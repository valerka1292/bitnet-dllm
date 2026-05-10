from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bitlinear import BitLinear
from .attention  import BitDiffAttention


class SwiGLUFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        H, F_dim = config.hidden_size, config.ffn_hidden_size
        kw = dict(activation_bits=config.activation_bits)
        self.gate_proj = BitLinear(H, F_dim, **kw)
        self.up_proj   = BitLinear(H, F_dim, **kw)
        self.down_proj = BitLinear(F_dim, H, **kw)
        self.drop      = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ste, eta, Q_b = self.gate_proj.quantize_input(x)
        gate = self.gate_proj.forward_quantized(x_ste, eta, Q_b)
        up   = self.up_proj.forward_quantized(x_ste, eta, Q_b)
        return self.down_proj(self.drop(F.silu(gate) * up))


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        normed = self.norm(x)
        if t_emb is None:
            return normed, None
        scale, shift, gate = self.proj(t_emb).unsqueeze(1).chunk(3, dim=-1)
        return normed * (1.0 + scale) + shift, gate


class BitDiffBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_ts = config.use_timestep_cond
        norm_cls    = AdaptiveRMSNorm if config.use_timestep_cond else nn.RMSNorm
        self.norm1  = norm_cls(config.hidden_size)
        self.norm2  = norm_cls(config.hidden_size)
        self.attn   = BitDiffAttention(config)
        self.ffn    = SwiGLUFFN(config)

    def _forward_attn(self, x, mask, t_emb):
        normed_attn, gate_attn = self.norm1(x, t_emb) if (self.use_ts and t_emb is not None) else (*self.norm1(x), None)
        attn_out = self.attn(normed_attn, mask)
        if gate_attn is not None:
            x = x + (1.0 + gate_attn) * attn_out
        else:
            x = x + attn_out
        return x

    def _forward_ffn(self, x, t_emb):
        normed_ffn, gate_ffn = self.norm2(x, t_emb) if (self.use_ts and t_emb is not None) else (*self.norm2(x), None)
        ffn_out = self.ffn(normed_ffn)
        if gate_ffn is not None:
            x = x + (1.0 + gate_ffn) * ffn_out
        else:
            x = x + ffn_out
        return x

    def forward(self, x: torch.Tensor, mask: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        if self.training and self.attn.training and getattr(self, 'gradient_checkpointing', False):
            x = torch.utils.checkpoint.checkpoint(self._forward_attn, x, mask, t_emb, use_reentrant=False)
            x = torch.utils.checkpoint.checkpoint(self._forward_ffn, x, t_emb, use_reentrant=False)
        else:
            x = self._forward_attn(x, mask, t_emb)
            x = self._forward_ffn(x, t_emb)
        return x
