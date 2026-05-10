from __future__ import annotations
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pydantic import BaseModel, ConfigDict, Field
from accelerate import Accelerator

from .model     import BitDiffLM
from .loss      import BitDiffLMLoss
from .tracker   import Tracker, ConsoleTracker
from .utils     import count_parameters


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate:   float  = 3e-4
    batch_size:      int    = 32
    num_epochs:      int    = 10
    gradient_clip:   float  = 1.0
    log_every:       int    = 100
    save_every:      int    = 1000
    save_dir:        str    = "./checkpoints"
    num_workers:     int    = 4
    grad_accum:      int    = Field(1, ge=1)
    ema_decay:       float  = 0.9999
    weight_decay:    float  = 0.01
    warmup_ratio:    float  = 0.05
    min_lr_ratio:    float  = 0.01
    betas:           tuple  = (0.9, 0.95)
    eps:             float  = 1e-8
    gradient_checkpointing: bool = False


class BitDiffLMTrainer:
    def __init__(
        self,
        model:         BitDiffLM,
        optimizer:     torch.optim.Optimizer,
        scheduler:     torch.optim.lr_scheduler.LambdaLR,
        train_loader:  DataLoader,
        config:        TrainingConfig,
        val_loader:    DataLoader | None = None,
        tracker:       Tracker | None = None,
    ):
        self.config     = config
        self.tracker    = tracker or ConsoleTracker()
        self.global_step = 0
        self.loss_fn    = BitDiffLMLoss(
            mask_token_id=model.config.mask_token_id,
            t_min=model.config.t_min,
            time_eps=model.config.time_eps,
        )

        self.accelerator = Accelerator(gradient_accumulation_steps=config.grad_accum)
        self.model, self.optimizer, self.train_loader, self.scheduler = \
            self.accelerator.prepare(model, optimizer, train_loader, scheduler)
        self.val_loader = self.accelerator.prepare(val_loader) if val_loader else None

        if config.gradient_checkpointing:
            raw = self.accelerator.unwrap_model(self.model)
            raw.config.gradient_checkpointing = True
            for b in raw.blocks:
                b.gradient_checkpointing = True

        raw = self.accelerator.unwrap_model(self.model)
        self.ema_model = BitDiffLM(raw.config).to(device=raw.device).eval()
        self.ema_model.load_state_dict(raw.state_dict())
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        self._log_info()

    def _log_info(self):
        raw = self.accelerator.unwrap_model(self.model)
        s   = count_parameters(raw)
        lines = [
            "─" * 56,
            f"  total params:  {s['total']:,}  |  ternary: {s['ternary']:,}  |  float: {s['float']:,}",
            f"  inference:     {s['inference_mb']:.1f} MB  |  training: {s['training_mb']:.1f} MB",
        ]
        if raw.config.max_seq_len > 512:
            lines.append(f"  ⚠  Flash Attention required for seq_len={raw.config.max_seq_len}")
        lines.append(f"  epochs: {self.config.num_epochs}  grad_accum: {self.config.grad_accum}")
        lines.append("─" * 56)
        self.tracker.log_line("\n".join(lines))

    def _update_ema(self):
        raw_model = self.accelerator.unwrap_model(self.model)
        decay = self.config.ema_decay
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), raw_model.parameters()):
                ema_p.data.mul_(decay).add_(model_p.data, alpha=1.0 - decay)

    def train_step(self, batch: dict) -> dict:
        self.model.train()
        out = self.model(batch["input_ids"], batch["attention_mask"], batch["timestep"])
        lo  = self.loss_fn(out["logits"], batch["labels"], batch["input_ids"], batch["timestep"], batch["attention_mask"])
        return lo

    @torch.no_grad()
    def validate(self) -> dict:
        if not self.val_loader:
            return {}
        self.model.eval()
        total_nll, total_n = 0.0, 0
        for batch in self.val_loader:
            out = self.model(batch["input_ids"], batch["attention_mask"], batch["timestep"])
            lo  = self.loss_fn(out["logits"], batch["labels"], batch["input_ids"], batch["timestep"], batch["attention_mask"])
            n = lo["n_masked"]
            loss_unwt = lo["loss_unweighted"] * n
            n_tensor = torch.tensor(n, device=self.accelerator.device, dtype=torch.float)
            gathered_loss, gathered_n = self.accelerator.gather_for_metrics((torch.tensor(loss_unwt, device=self.accelerator.device), n_tensor))
            total_nll += gathered_loss.sum().item()
            total_n   += gathered_n.sum().item()
        avg = total_nll / max(total_n, 1)
        return {"val_nll": avg, "val_ppl": math.exp(min(avg, 20))}

    def save_checkpoint(self, path: str | Path):
        tmp_path = str(path) + ".tmp"
        self.accelerator.save_state(tmp_path)
        os.replace(tmp_path, path)

    def load_checkpoint(self, path: str | Path):
        self.accelerator.load_state(path)

    def save_pretrained(self, save_dir: str | Path):
        save_dir = Path(save_dir)
        raw = self.accelerator.unwrap_model(self.model)
        raw.save_pretrained(save_dir)
        self.ema_model.save_pretrained(save_dir / "ema")
        self.save_checkpoint(save_dir / "trainer.pt")

    def train(self, num_epochs: int | None = None):
        n_epochs = num_epochs or self.config.num_epochs
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(n_epochs):
            losses = []
            for batch in self.train_loader:
                with self.accelerator.accumulate(self.model):
                    lo = self.train_step(batch)
                    self.accelerator.backward(lo["loss"])

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if self.accelerator.sync_gradients:
                        self._update_ema()
                        self.global_step += 1

                if self.global_step % self.config.log_every == 0 and self.accelerator.sync_gradients:
                    self.tracker.log({
                        "epoch":     epoch + 1,
                        "step":      self.global_step,
                        "loss":      lo["loss"],
                        "loss_unwt": lo["loss_unweighted"],
                        "n_masked":  lo["n_masked"],
                        "lr":        self.scheduler.get_last_lr()[0],
                    }, step=self.global_step)

                if self.global_step % self.config.save_every == 0 and self.accelerator.sync_gradients:
                    ckpt_dir = save_dir / f"step_{self.global_step}"
                    raw = self.accelerator.unwrap_model(self.model)
                    raw.save_pretrained(ckpt_dir)
                    self.ema_model.save_pretrained(ckpt_dir / "ema")
                    self.save_checkpoint(ckpt_dir / "trainer.pt")

                if not self.accelerator.sync_gradients:
                    losses.append(lo["loss"].item() if torch.is_tensor(lo["loss"]) else lo["loss"])

            val = self.validate()
            local_losses = [l for l in losses] if losses else [0.0]
            avg = sum(local_losses) / len(local_losses)
            if val:
                self.tracker.log({"epoch": epoch + 1, "n_epochs": n_epochs, "loss": avg, "val_nll": val["val_nll"], "val_ppl": val["val_ppl"]})
            else:
                self.tracker.log({"epoch": epoch + 1, "n_epochs": n_epochs, "loss": avg})

        raw = self.accelerator.unwrap_model(self.model)
        raw.save_pretrained(save_dir / "final")
        self.save_checkpoint(save_dir / "final" / "trainer.pt")
