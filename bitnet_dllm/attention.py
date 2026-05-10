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

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        L_q = q.shape[2]
        L_k = k.shape[2]
        need_len = max(L_k, L_q + offset)
        device = q.device
        if need_len <= self.max_seq_len:
            if not hasattr(self, "_cos") or self._cos.shape[-1] < need_len:
                self._build_cache(self.max_seq_len)
            cos_k = self._cos[:, :, :L_k].to(device=device, dtype=q.dtype)
            sin_k = self._sin[:, :, :L_k].to(device=device, dtype=q.dtype)
        else:
            warnings.warn(
                f"total_seq_len={need_len} > max_seq_len={self.max_seq_len}. RoPE extrapolating.",
                UserWarning, stacklevel=3,
            )
            if not hasattr(self, "_cos") or self._cos.shape[-1] < need_len:
                self._build_cache(need_len)
            cos_k = self._cos[:, :, :L_k].to(device=device, dtype=q.dtype)
            sin_k = self._sin[:, :, :L_k].to(device=device, dtype=q.dtype)
        cos_q = self._cos[:, :, offset:offset + L_q].to(device=device, dtype=q.dtype)
        sin_q = self._sin[:, :, offset:offset + L_q].to(device=device, dtype=q.dtype)
        return q * cos_q + self._rotate_half(q) * sin_q, k * cos_k + self._rotate_half(k) * sin_k


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

    def forward(
        self,
        x:                torch.Tensor,
        attention_mask:   torch.Tensor,
        past_key_values:  tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, L, D = x.shape
        H, Dh   = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, Dh).transpose(1, 2)

        kv_offset = 0
        if past_key_values is not None:
            k_cache, v_cache = past_key_values
            kv_offset = k_cache.shape[2]
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        present_key_values = (k, v)

        q, k  = self.rotary(q, k, offset=kv_offset)

        if attention_mask is not None:
            if past_key_values is not None:
                cache_mask = attention_mask.new_ones(B, k_cache.shape[2])
                full_mask = torch.cat([cache_mask, attention_mask], dim=1)
            else:
                full_mask = attention_mask
            bool_mask = full_mask.bool().unsqueeze(1).unsqueeze(2)
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
        return self.out_proj(out), present_key_values
