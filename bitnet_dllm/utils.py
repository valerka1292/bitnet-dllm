from __future__ import annotations


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
    return {"total": bit + fp, "ternary": bit, "float": fp}


def print_model_info(model):
    cfg = model.config
    s   = model.memory_stats()
    print("─" * 56)
    print(f"  BitDiffLM  hidden={cfg.hidden_size}  layers={cfg.num_layers}  heads={cfg.num_heads}")
    print(f"  ffn_hidden={cfg.ffn_hidden_size}  seq_len={cfg.max_seq_len}  vocab={cfg.vocab_size}")
    print(f"  total:     {s['total']:>12,}")
    print(f"  ternary:   {s['ternary']:>12,}")
    print(f"  float:     {s['float']:>12,}")
    print(f"  inference: {s['inference_mb']:>8.1f} MB")
    print(f"  training:  {s['training_mb']:>8.1f} MB  (params only)")
    if s['flash_required']:
        print(f"  ⚠  seq_len={cfg.max_seq_len}: Flash Attention strongly recommended")
    print("─" * 56)
