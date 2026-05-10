from __future__ import annotations
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from .bitlinear import BitLinear


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.head_dim    = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def _build_cache(self, seq_len: int):
        device = self.inv_freq.device
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("_cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("_sin", emb.sin()[None, None], persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        L = q.shape[2]
        device = q.device
        if L <= self.max_seq_len:
            if not hasattr(self, "_cos"):
                self._build_cache(self.max_seq_len)
            cos = self._cos[:, :, :L].to(device=device, dtype=q.dtype)
            sin = self._sin[:, :, :L].to(device=device, dtype=q.dtype)
        else:
            warnings.warn(
                f"seq_len={L} > max_seq_len={self.max_seq_len}. RoPE extrapolating.",
                UserWarning, stacklevel=3,
            )
            if not hasattr(self, "_cos") or self._cos.shape[-1] < L:
                self._build_cache(L)
            cos = self._cos[:, :, :L].to(device=device, dtype=q.dtype)
            sin = self._sin[:, :, :L].to(device=device, dtype=q.dtype)
        return q * cos + self._rotate_half(q) * sin, k * cos + self._rotate_half(k) * sin


class BitDiffAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim  = config.head_dim
        self.scale     = config.head_dim ** -0.5
        H, kw = config.hidden_size, dict(activation_bits=config.activation_bits)
        self.q_proj   = BitLinear(H, H, **kw)
        self.k_proj   = BitLinear(H, H, **kw)
        self.v_proj   = BitLinear(H, H, **kw)
        self.out_proj = BitLinear(H, H, **kw)
        self.rotary   = RotaryEmbedding(config.head_dim, config.max_seq_len)
        self.dropout  = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, Dh).transpose(1, 2)

        q, k  = self.rotary(q, k)

        if attention_mask is not None:
            bool_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)
        else:
            bool_mask = None

        p_drop = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=bool_mask,
            dropout_p=p_drop,
            is_causal=False
        )

        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)
