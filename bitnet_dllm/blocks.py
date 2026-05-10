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

    def forward(self, x: torch.Tensor, mask: torch.Tensor, t_emb: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_ts and t_emb is not None:
            normed_attn, gate_attn = self.norm1(x, t_emb)
            x = x + (1.0 + gate_attn) * self.attn(normed_attn, mask)
            normed_ffn, gate_ffn = self.norm2(x, t_emb)
            x = x + (1.0 + gate_ffn) * self.ffn(normed_ffn)
        else:
            normed_attn, _ = self.norm1(x)
            x = x + self.attn(normed_attn, mask)
            normed_ffn, _ = self.norm2(x)
            x = x + self.ffn(normed_ffn)
        return x
