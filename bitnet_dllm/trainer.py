from __future__ import annotations
import math
import os
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .model     import BitDiffLM
from .loss      import BitDiffLMLoss
from .dataset   import MaskedDiffusionDataset, worker_init_fn


class BitDiffLMTrainer:
    def __init__(
        self,
        model:          BitDiffLM,
        train_dataset:  MaskedDiffusionDataset,
        val_dataset:    MaskedDiffusionDataset | None = None,
        batch_size:     int   = 32,
        learning_rate:  float = 3e-4,
        num_epochs:     int   = 10,
        gradient_clip:  float = 1.0,
        log_every:      int   = 100,
        save_every:     int   = 1000,
        save_dir:       str   = "./checkpoints",
        device:         str   = "cuda",
        num_workers:    int   = 4,
        grad_accum:     int   = 1,
    ):
        self.model        = model.to(device)
        self.device       = device
        self.num_epochs   = num_epochs
        self.grad_clip    = gradient_clip
        self.log_every    = log_every
        self.save_every   = save_every
        self.save_dir     = Path(save_dir)
        self.grad_accum   = grad_accum
        self.global_step  = 0

        cfg = model.config

        ld_kw = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(device == "cuda"),
            worker_init_fn=worker_init_fn,
        )
        self.train_loader = DataLoader(train_dataset, shuffle=True,  collate_fn=train_dataset.get_collate_fn(), **ld_kw)
        self.val_loader   = DataLoader(val_dataset,   shuffle=False, collate_fn=val_dataset.get_collate_fn(),   **ld_kw) if val_dataset else None

        self.loss_fn = BitDiffLMLoss(mask_token_id=cfg.mask_token_id, t_min=cfg.t_min)

        total_steps = (len(self.train_loader) // grad_accum) * num_epochs
        optim_cfg = model.configure_optimizers(
            learning_rate=learning_rate,
            weight_decay=cfg.weight_decay,
            total_steps=total_steps,
            warmup_ratio=cfg.warmup_ratio,
            min_lr_ratio=cfg.min_lr_ratio,
        )
        self.optimizer = optim_cfg["optimizer"]
        self.scheduler = optim_cfg["scheduler"]

        self._print_info(optim_cfg["groups"], total_steps, optim_cfg["warmup_steps"])

    def _print_info(self, groups, total_steps, warmup_steps):
        s = self.model.memory_stats()
        print("─" * 56)
        print(f"  total params:  {s['total']:,}  |  ternary: {s['ternary']:,}  |  float: {s['float']:,}")
        print(f"  inference:     {s['inference_mb']:.1f} MB  |  training: {s['training_mb']:.1f} MB")
        if s['flash_required']:
            print(f"  ⚠  Flash Attention required for seq_len={self.model.config.max_seq_len}")
        print(f"  steps: {total_steps}  warmup: {warmup_steps}  grad_accum: {self.grad_accum}")
        for k, v in groups.items():
            print(f"  [{k}] {sum(p.numel() for p in v):,} params")
        print("─" * 56)

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        ids  = batch["input_ids"].to(self.device)
        lbl  = batch["labels"].to(self.device)
        attn = batch["attention_mask"].to(self.device)
        t    = batch["timestep"].to(self.device)
        out  = self.model(ids, attn, t)
        lo   = self.loss_fn(out["logits"], lbl, ids, t, attn)
        (lo["loss"] / self.grad_accum).backward()
        return lo

    @torch.no_grad()
    def validate(self) -> dict:
        if not self.val_loader:
            return {}
        self.model.eval()
        total_nll, total_n = 0.0, 0
        for batch in self.val_loader:
            ids  = batch["input_ids"].to(self.device)
            lbl  = batch["labels"].to(self.device)
            attn = batch["attention_mask"].to(self.device)
            t    = batch["timestep"].to(self.device)
            out  = self.model(ids, attn, t)
            lo   = self.loss_fn(out["logits"], lbl, ids, t, attn)
            n = lo["n_masked"]
            total_nll += lo["loss_unweighted"] * n
            total_n   += n
        avg = total_nll / max(total_n, 1)
        return {"val_nll": avg, "val_ppl": math.exp(min(avg, 20))}

    def save_checkpoint(self, path: str | Path):
        tmp_path = str(path) + ".tmp"
        torch.save({
            "global_step": self.global_step,
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
        }, tmp_path)
        os.replace(tmp_path, path)

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location="cpu")
        self.global_step = ckpt["global_step"]
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])

    def save_pretrained(self, save_dir: str | Path):
        self.model.save_pretrained(save_dir)
        self.save_checkpoint(Path(save_dir) / "trainer.pt")

    def train(self, num_epochs: int | None = None):
        n_epochs = num_epochs or self.num_epochs
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer.zero_grad()

        ema_model = deepcopy(self.model).eval()
        ema_decay = 0.9999

        for epoch in range(n_epochs):
            losses = []
            for batch_idx, batch in enumerate(self.train_loader):
                lo = self.train_step(batch)
                losses.append(lo["loss"].item() if torch.is_tensor(lo["loss"]) else lo["loss"])

                if (batch_idx + 1) % self.grad_accum == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.scheduler.step()
                    with torch.no_grad():
                        for param_q, param_ema in zip(self.model.parameters(), ema_model.parameters()):
                            param_ema.data.mul_(ema_decay).add_(param_q.data, alpha=1.0 - ema_decay)
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % self.log_every == 0:
                        print(
                            f"[e{epoch+1} s{self.global_step}] "
                            f"loss={lo['loss']:.4f} unwt={lo['loss_unweighted']:.4f} "
                            f"masked={lo['n_masked']} lr={self.scheduler.get_last_lr()[0]:.2e}"
                        )
                    if self.global_step % self.save_every == 0:
                        ckpt_dir = self.save_dir / f"step_{self.global_step}"
                        self.model.save_pretrained(ckpt_dir)
                        self.save_checkpoint(ckpt_dir / "trainer.pt")

            val = self.validate()
            avg = sum(losses) / len(losses)
            val_str = f"val_nll={val['val_nll']:.4f}  ppl={val['val_ppl']:.1f}" if val else "no val"
            print(f"\n=== epoch {epoch+1}/{n_epochs}  loss={avg:.4f}  {val_str} ===\n")

        self.model.save_pretrained(self.save_dir / "final")
        self.save_checkpoint(self.save_dir / "final" / "trainer.pt")
