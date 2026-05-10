from __future__ import annotations
import math
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from .config    import BitDiffLMConfig
from .bitlinear import BitLinear
from .blocks    import BitDiffBlock, AdaptiveRMSNorm


class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        half  = freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
        self.register_buffer("freqs", freqs, persistent=True)
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None].float() * self.freqs[None, :]
        emb  = torch.cat([args.sin(), args.cos()], dim=-1)
        return self.norm(self.mlp(emb))


class BitDiffLM(nn.Module):
    def __init__(self, config: BitDiffLMConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.ts_emb    = TimestepEmbedding(config.hidden_size, config.timestep_freq_dim) if config.use_timestep_cond else None
        self.blocks    = nn.ModuleList([BitDiffBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.RMSNorm(config.hidden_size)
        self.lm_head   = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        ada_proj_ids   = {id(m.proj) for m in self.modules() if isinstance(m, AdaptiveRMSNorm)}
        tied_w_ids     = {id(self.lm_head.weight)} if self.config.tie_word_embeddings else set()

        for module in self.modules():
            if isinstance(module, BitLinear):
                nn.init.normal_(module.weight, 0.0, 1.0 / math.sqrt(module.weight.shape[1]))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.Linear):
                if id(module) in ada_proj_ids or id(module.weight) in tied_w_ids:
                    continue
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        timestep:       torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x     = self.token_emb(input_ids)
        t_emb = self.ts_emb(timestep) if (self.ts_emb is not None and timestep is not None) else None
        for block in self.blocks:
            x = block(x, attention_mask, t_emb)
        x      = self.final_norm(x)
        logits = self.lm_head(x)
        return {"logits": logits, "hidden_states": x}

    def no_weight_decay_parameters(self) -> set[str]:
        norm_ids  = {id(m) for m in self.modules() if isinstance(m, (nn.RMSNorm, nn.LayerNorm))}

        param_to_parent: dict[str, int] = {}
        for mod_name, mod in self.named_modules():
            for p_name, param in mod.named_parameters(recurse=False):
                full = f"{mod_name}.{p_name}" if mod_name else p_name
                param_to_parent[full] = id(mod)

        no_decay: set[str] = set()
        for name, param in self.named_parameters():
            if name.endswith(".bias") or param_to_parent.get(name) in norm_ids:
                no_decay.add(name)
        return no_decay

    def save_pretrained(self, save_dir: str | Path):
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.config.save(d / "config.json")
        torch.save(self.state_dict(), d / "model.pt")

    @classmethod
    def from_pretrained(cls, save_dir: str | Path, device: str = "cpu") -> BitDiffLM:
        d      = Path(save_dir)
        config = BitDiffLMConfig.load(d / "config.json")
        model  = cls(config)
        state  = torch.load(d / "model.pt", map_location="cpu")
        missing, unexpected = (
            set(model.state_dict()) - set(state),
            set(state) - set(model.state_dict()),
        )
        if missing or unexpected:
            warnings.warn(f"state_dict mismatch: missing={missing}, unexpected={unexpected}")
        model.load_state_dict(state, strict=False)
        return model.to(device)

    def memory_stats(self) -> dict:
        return self.config.memory_stats()
