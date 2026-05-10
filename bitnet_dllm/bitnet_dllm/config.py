from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class BitDiffLMConfig:
    vocab_size:          int   = 32000
    hidden_size:         int   = 768
    num_layers:          int   = 12
    num_heads:           int   = 12
    ffn_hidden_size:     int   = 2048
    max_seq_len:         int   = 1024
    dropout:             float = 0.0
    activation_bits:     int   = 8

    mask_token_id:       int   = 103
    pad_token_id:        int   = 0
    bos_token_id:        int   = 1
    eos_token_id:        int   = 2

    noise_schedule:      str   = "log_linear"
    t_min:               float = 0.05
    t_max:               float = 0.99

    use_timestep_cond:   bool  = True
    timestep_freq_dim:   int   = 256

    min_lr_ratio:        float = 0.1
    warmup_ratio:        float = 0.05
    weight_decay:        float = 0.01

    tie_word_embeddings: bool  = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def validate(self):
        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size={self.hidden_size} не делится на num_heads={self.num_heads}"
        assert self.activation_bits in (4, 8)
        assert 0 < self.t_min < self.t_max <= 1.0
        assert self.vocab_size > max(
            self.mask_token_id, self.pad_token_id,
            self.bos_token_id, self.eos_token_id
        )

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> BitDiffLMConfig:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        known = {f for f in cls.__dataclass_fields__}
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        cfg.validate()
        return cfg

    @classmethod
    def from_preset(cls, name: str, **overrides) -> BitDiffLMConfig:
        if name not in PRESETS:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS)}")
        return cls(**{**PRESETS[name], **overrides})

    def memory_stats(self) -> dict:
        H, F, N, V, L = (
            self.hidden_size, self.ffn_hidden_size,
            self.num_layers, self.vocab_size, self.max_seq_len,
        )
        bit_params  = N * (4 * H * H + 2 * H * F + F * H)
        emb_params  = V * H
        lm_h_params = 0 if self.tie_word_embeddings else V * H
        rms_params  = (N * 4 + 1) * H
        ada_params  = N * 2 * (H * 2 * H + 2 * H + 1) if self.use_timestep_cond else 0
        freq        = self.timestep_freq_dim
        ts_params   = (freq * 2 * H + 2 * H) + (2 * H * H + H) + H if self.use_timestep_cond else 0
        float_params = emb_params + lm_h_params + rms_params + ada_params + ts_params
        total        = bit_params + float_params
        return {
            "total":            total,
            "ternary":          bit_params,
            "float":            float_params,
            "inference_mb":     (bit_params * 1 + float_params * 2) / 1e6,
            "training_mb":      total * 16 / 1e6,
            "attn_act_mb":      N * self.num_heads * L * L * 4 / 1e6,
            "flash_required":   L > 512,
        }


PRESETS: dict[str, dict] = {
    "tiny":  dict(hidden_size=256,  num_layers=4,  num_heads=4,  ffn_hidden_size=683,  max_seq_len=256),
    "small": dict(hidden_size=768,  num_layers=12, num_heads=12, ffn_hidden_size=2048, max_seq_len=1024),
    "base":  dict(hidden_size=1024, num_layers=24, num_heads=16, ffn_hidden_size=2731, max_seq_len=2048),
    "large": dict(hidden_size=1536, num_layers=32, num_heads=24, ffn_hidden_size=4096, max_seq_len=4096),
}
