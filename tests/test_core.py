from __future__ import annotations
import math
import warnings
import pytest
import torch
import torch.nn as nn

from bitnet_dllm import BitDiffLMConfig, BitDiffLM, BitDiffLMLoss, MaskedDiffusionDataset
from bitnet_dllm.bitlinear import BitLinear
from bitnet_dllm.blocks    import AdaptiveRMSNorm
from bitnet_dllm.utils     import get_parameter_groups


@pytest.fixture
def cfg():
    return BitDiffLMConfig(
        vocab_size=200, hidden_size=64, num_layers=2, num_heads=4,
        ffn_hidden_size=171, max_seq_len=32, mask_token_id=1,
        pad_token_id=0, use_timestep_cond=True, dropout=0.0,
    )

@pytest.fixture
def model(cfg):
    return BitDiffLM(cfg)

@pytest.fixture
def batch(cfg):
    B, L = 4, 32
    ids  = torch.randint(2, cfg.vocab_size, (B, L))
    lbl  = ids.clone()
    m    = torch.rand(B, L) < 0.3
    ids[m] = cfg.mask_token_id
    return {
        "input_ids":      ids,
        "labels":         lbl,
        "attention_mask": torch.ones(B, L, dtype=torch.long),
        "timestep":       torch.rand(B) * 0.8 + 0.1,
    }


def test_gradient_flow(model, batch, cfg):
    loss_fn = BitDiffLMLoss(mask_token_id=cfg.mask_token_id, t_min=cfg.t_min)
    out  = model(batch["input_ids"], batch["attention_mask"], batch["timestep"])
    loss = loss_fn(out["logits"], batch["labels"], batch["input_ids"], batch["timestep"], batch["attention_mask"])["loss"]
    loss.backward()

    # ts_emb params and AdaptiveRMSNorm.proj are zero-initialized and block
    # gradient flow to timestep encoder at init — skip them.
    skip_prefixes = ("ts_emb.",) + tuple(
        f"blocks.{i}.norm{j}.proj." for i in range(cfg.num_layers) for j in (1, 2)
    )
    failed = [
        n for n, p in model.named_parameters()
        if not any(n.startswith(sp) for sp in skip_prefixes)
        and (p.grad is None or p.grad.norm() < 1e-12)
    ]
    assert not failed, f"No/zero gradient: {failed[:5]}"


def test_ste_input_gradient():
    layer = BitLinear(32, 32, activation_bits=8)
    x     = torch.randn(2, 4, 32, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None and x.grad.norm() > 1e-10


def test_swiglu_shared_quantization(cfg):
    from bitnet_dllm.blocks import SwiGLUFFN
    ffn = SwiGLUFFN(cfg)
    x   = torch.randn(2, 8, cfg.hidden_size, requires_grad=True)
    out = ffn(x)
    out.sum().backward()
    assert x.grad is not None


def test_adaptive_rms_norm_none():
    norm     = AdaptiveRMSNorm(64)
    x        = torch.randn(2, 8, 64)
    out_none = norm(x, t_emb=None)
    out_rms  = norm.norm(x)
    assert isinstance(out_none, tuple) and out_none[1] is None
    assert torch.allclose(out_none[0], out_rms, atol=1e-6)


def test_multinomial_safe():
    logits = torch.full((8, 100), float("-inf"))
    logits[:, 0] = 1.0
    probs  = torch.softmax(logits, dim=-1).clamp(min=1e-10)
    probs /= probs.sum(-1, keepdim=True)
    samp   = torch.multinomial(probs, 1)
    assert (samp == 0).all()


def test_no_weight_decay(model, cfg):
    no_decay = model.no_weight_decay_parameters()

    for name, mod in model.named_modules():
        if isinstance(mod, (nn.RMSNorm, nn.LayerNorm)):
            for pn, p in mod.named_parameters(recurse=False):
                assert f"{name}.{pn}" in no_decay, f"{name}.{pn} should be in no_decay"

    for name, mod in model.named_modules():
        if isinstance(mod, BitLinear):
            assert f"{name}.weight" not in no_decay

    for name, mod in model.named_modules():
        if isinstance(mod, AdaptiveRMSNorm):
            assert f"{name}.proj.weight" not in no_decay, f"proj.weight should have decay"


def test_save_load(model, batch, tmp_path):
    model.eval()
    with torch.no_grad():
        before = model(batch["input_ids"], batch["attention_mask"], batch["timestep"])["logits"]
    model.save_pretrained(tmp_path)
    loaded = BitDiffLM.from_pretrained(tmp_path)
    loaded.eval()
    with torch.no_grad():
        after = loaded(batch["input_ids"], batch["attention_mask"], batch["timestep"])["logits"]
    assert torch.allclose(before, after, atol=1e-5)
    assert loaded.config.mask_token_id == model.config.mask_token_id


def test_config_extra_forbid():
    with pytest.raises(ValueError, match="extra"):
        BitDiffLMConfig(dim=128, hidden_size=64)

    with pytest.raises(ValueError, match="extra"):
        BitDiffLMConfig(hidden_size=64, num_layers=2, unknown_param=True)


def test_get_parameter_groups_returns_only_lists(cfg):
    model = BitDiffLM(cfg)
    groups = get_parameter_groups(model, learning_rate=1e-3, weight_decay=0.01)
    assert isinstance(groups, list)
    assert len(groups) == 4
    for g in groups:
        assert "params" in g
        assert "lr" in g
        assert "weight_decay" in g
        assert isinstance(g["params"], list)


def test_ste_custom_function():
    layer = BitLinear(32, 32, activation_bits=8)
    x     = torch.randn(2, 4, 32, requires_grad=True)
    layer(x).sum().backward()
    assert x.grad is not None and x.grad.norm() > 1e-10


def test_gradient_checkpointing(cfg):
    cfg.gradient_checkpointing = True
    model = BitDiffLM(cfg)
    for b in model.blocks:
        assert b.gradient_checkpointing == True

    B, L = 2, 16
    ids  = torch.randint(2, cfg.vocab_size, (B, L))
    attn = torch.ones(B, L, dtype=torch.long)
    t    = torch.rand(B)
    out  = model(ids, attn, t)
    loss = out["logits"].sum()
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_loss_decreases(cfg):
    model   = BitDiffLM(cfg)
    loss_fn = BitDiffLMLoss(mask_token_id=cfg.mask_token_id, t_min=cfg.t_min)
    opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
    B, L    = 8, 16
    ids     = torch.randint(2, cfg.vocab_size, (B, L))
    lbl     = ids.clone()
    ids[:, ::3] = cfg.mask_token_id
    attn    = torch.ones(B, L, dtype=torch.long)
    t       = torch.full((B,), 0.5)
    losses  = []
    for _ in range(30):
        out = model(ids, attn, t)
        lo  = loss_fn(out["logits"], lbl, ids, t, attn)["loss"]
        opt.zero_grad(); lo.backward(); opt.step()
        losses.append(lo.item())
    assert losses[-1] < losses[0], f"Loss not decreasing: {losses[0]:.4f} → {losses[-1]:.4f}"


def test_sampler_reaches_zero(cfg):
    from bitnet_dllm.sampler import MDLMAncestralSampler
    import types

    model = BitDiffLM(cfg)

    class FakeTok:
        mask_token_id = cfg.mask_token_id
        def decode(self, ids, skip_special_tokens=True):
            return " ".join(map(str, ids))

    sampler   = MDLMAncestralSampler(model, FakeTok(), device="cpu")
    t_steps   = torch.linspace(1.0, 0.0, 11)
    assert t_steps[-1].item() == 0.0, "t_steps must reach 0"

    steps_seen = []
    def cb(step, total, t, decoded):
        steps_seen.append(t)

    result = sampler.generate(seq_len=16, num_steps=10, batch_size=2, step_callback=cb)
    assert len(result) == 2
    assert len(steps_seen) == 10


def test_dataset_collate_masking(cfg):
    seqs    = [torch.randint(2, 100, (20,)) for _ in range(40)]
    dataset = MaskedDiffusionDataset(seqs, mask_token_id=cfg.mask_token_id, t_min=0.1, t_max=0.9)
    item    = dataset[0]
    assert isinstance(item["input_ids"], (torch.Tensor, list))

    col = dataset.get_collate_fn()
    batch = col([dataset[i] for i in range(4)])
    assert batch["labels"][batch["attention_mask"] == 0].eq(-100).all()
    assert (batch["input_ids"] == cfg.mask_token_id).any()
    assert batch["input_ids"].shape[0] == 4


def test_rope_extends_beyond_max_seq_len(cfg):
    model = BitDiffLM(cfg)
    model.eval()
    L    = cfg.max_seq_len * 2
    ids  = torch.randint(2, cfg.vocab_size, (1, L))
    attn = torch.ones(1, L)
    t    = torch.tensor([0.5])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with torch.no_grad():
            out = model(ids, attn, t)
        assert any("max_seq_len" in str(x.message) for x in w)
    assert out["logits"].shape == (1, L, cfg.vocab_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_pipeline(cfg, tmp_path):
    model = BitDiffLM(cfg)
    model.save_pretrained(tmp_path)
    model_c = BitDiffLM.from_pretrained(tmp_path, device="cuda")
    model_c.eval()
    ids  = torch.randint(2, cfg.vocab_size, (2, 16)).cuda()
    attn = torch.ones(2, 16).cuda()
    t    = torch.rand(2).cuda()
    with torch.no_grad():
        out = model_c(ids, attn, t)
    assert out["logits"].device.type == "cuda"
