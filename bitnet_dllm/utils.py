from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def _cosine_schedule(warmup: int, total: int, min_ratio: float):
    def fn(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * p)))
    return fn


def get_optimizer_groups(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    total_steps: int,
    warmup_ratio: float = 0.05,
    min_lr_ratio: float = 0.01,
    betas: tuple = (0.9, 0.95),
    eps: float = 1e-8,
) -> dict:
    from .bitlinear import BitLinear

    for m in model.modules():
        if isinstance(m, (nn.RMSNorm, nn.LayerNorm)):
            if hasattr(m, 'weight') and m.weight is not None:
                m.weight._no_weight_decay = True
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias._no_weight_decay = True

    bit_ids = {id(m) for m in model.modules() if isinstance(m, BitLinear)}

    param_parent: dict[str, int] = {}
    for mn, m in model.named_modules():
        for pn, _ in m.named_parameters(recurse=False):
            param_parent[f"{mn}.{pn}" if mn else pn] = id(m)

    groups: dict[str, list] = {"bit_d": [], "bit_nd": [], "fp_d": [], "fp_nd": []}
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        is_bit = param_parent.get(name) in bit_ids
        is_nd = getattr(param, '_no_weight_decay', False)
        key = ("bit" if is_bit else "fp") + ("_nd" if is_nd else "_d")
        groups[key].append(param)

    warmup_steps = int(total_steps * warmup_ratio)
    optimizer = AdamW([
        {"params": groups["bit_d"],  "lr": learning_rate * 0.7, "weight_decay": weight_decay},
        {"params": groups["bit_nd"], "lr": learning_rate * 0.7, "weight_decay": 0.0},
        {"params": groups["fp_d"],   "lr": learning_rate,       "weight_decay": weight_decay},
        {"params": groups["fp_nd"],  "lr": learning_rate,       "weight_decay": 0.0},
    ], betas=betas, eps=eps)
    scheduler = LambdaLR(optimizer, _cosine_schedule(warmup_steps, total_steps, min_lr_ratio))

    return {"optimizer": optimizer, "scheduler": scheduler, "groups": groups, "total_steps": total_steps, "warmup_steps": warmup_steps}


def count_parameters(model) -> dict:
    from .bitlinear import BitLinear
    bit, fp = 0, 0
    seen = set()
    for name, param in model.named_parameters():
        if id(param) in seen:
            continue
        seen.add(id(param))
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            try:
                parent = model
                for a in parts[0].split("."):
                    parent = getattr(parent, a)
                if isinstance(parent, BitLinear):
                    bit += param.numel()
                    continue
            except AttributeError:
                pass
        fp += param.numel()

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
