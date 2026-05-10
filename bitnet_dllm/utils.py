from __future__ import annotations

import math
import torch
import torch.nn as nn


def _cosine_schedule(warmup: int, total: int, min_ratio: float):
    def fn(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * p)))
    return fn


def get_parameter_groups(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> list[dict]:
    from .bitlinear import BitLinear

    for m in model.modules():
        if isinstance(m, (nn.RMSNorm, nn.LayerNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight._no_weight_decay = True
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias._no_weight_decay = True

    all_params = set(model.parameters())

    bit_params: set[torch.Tensor] = set()
    for m in model.modules():
        if isinstance(m, BitLinear):
            bit_params.update(m.parameters())

    fp_params = all_params - bit_params

    bit_d  = [p for p in bit_params if not getattr(p, '_no_weight_decay', False)]
    bit_nd = [p for p in bit_params if getattr(p, '_no_weight_decay', False)]
    fp_d   = [p for p in fp_params if not getattr(p, '_no_weight_decay', False)]
    fp_nd  = [p for p in fp_params if getattr(p, '_no_weight_decay', False)]

    return [
        {"params": bit_d,  "lr": learning_rate * 0.7, "weight_decay": weight_decay},
        {"params": bit_nd, "lr": learning_rate * 0.7, "weight_decay": 0.0},
        {"params": fp_d,   "lr": learning_rate,       "weight_decay": weight_decay},
        {"params": fp_nd,  "lr": learning_rate,       "weight_decay": 0.0},
    ]





def count_parameters(model) -> dict:
    from .bitlinear import BitLinear
    bit_params = set()
    for m in model.modules():
        if isinstance(m, BitLinear):
            bit_params.update(m.parameters())
    all_params = set(model.parameters())
    bit = sum(p.numel() for p in bit_params)
    fp = sum(p.numel() for p in (all_params - bit_params))
    total_params = bit + fp
    inference_mb = (bit * 1 + fp * 2) / 1e6
    training_mb = total_params * 16 / 1e6

    return {
        "total": total_params,
        "ternary": bit,
        "float": fp,
        "inference_mb": inference_mb,
        "training_mb": training_mb,
    }


def print_model_info(model):
    from .config import BitDiffLMConfig
    cfg = model.config
    s = count_parameters(model)
    print("─" * 56)
    print(f"  BitDiffLM  hidden={cfg.hidden_size}  layers={cfg.num_layers}  heads={cfg.num_heads}")
    print(f"  ffn_hidden={cfg.ffn_hidden_size}  seq_len={cfg.max_seq_len}  vocab={cfg.vocab_size}")
    print(f"  total:     {s['total']:>12,}")
    print(f"  ternary:   {s['ternary']:>12,}")
    print(f"  float:     {s['float']:>12,}")
    print(f"  inference: {s['inference_mb']:>8.1f} MB")
    print(f"  training:  {s['training_mb']:>8.1f} MB  (params only)")
    if cfg.max_seq_len > 512:
        print(f"  ⚠  seq_len={cfg.max_seq_len}: Flash Attention strongly recommended")
    print("─" * 56)
