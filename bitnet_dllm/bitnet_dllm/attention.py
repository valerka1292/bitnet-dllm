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
        self.register_buffer("_cached_len", torch.tensor(0, dtype=torch.long), persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        dev = self.inv_freq.device
        if seq_len > self.max_seq_len:
            t = torch.arange(seq_len, device=dev, dtype=self.inv_freq.dtype)
            t = t * (self.max_seq_len / seq_len)
        else:
            t = torch.arange(seq_len, device=dev, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("_cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("_sin", emb.sin()[None, None], persistent=False)
        self._cached_len.fill_(seq_len)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        L = q.shape[2]
        if L > self._cached_len.item() or self._cos.device != q.device or self._cos.dtype != q.dtype:
            if L > self.max_seq_len:
                warnings.warn(
                    f"seq_len={L} > max_seq_len={self.max_seq_len}. RoPE extrapolating.",
                    UserWarning, stacklevel=3,
                )
            self._build_cache(max(L, self.max_seq_len))
        cos = self._cos[:, :, :L].to(dtype=q.dtype)
        sin = self._sin[:, :, :L].to(dtype=q.dtype)
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
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            scores = scores - (1.0 - attention_mask.float())[:, None, None, :] * 1e9
        attn = self.dropout(F.softmax(scores, dim=-1))
        out  = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)
