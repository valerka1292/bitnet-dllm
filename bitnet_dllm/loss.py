from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class BitDiffLMLoss(nn.Module):
    def __init__(self, mask_token_id: int, t_min: float = 0.05):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.t_min         = t_min

    def forward(
        self,
        logits:         torch.Tensor,
        labels:         torch.Tensor,
        input_ids:      torch.Tensor,
        timestep:       torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        B, L, V = logits.shape

        is_valid = (
            (input_ids == self.mask_token_id) &
            attention_mask.bool() &
            (labels != -100)
        )

        if not is_valid.any():
            return {"loss": logits.sum() * 0.0, "n_masked": 0, "loss_unweighted": 0.0}

        ce = F.cross_entropy(
            logits.view(B * L, V),
            labels.view(B * L),
            reduction="none",
            ignore_index=-100,
        ).view(B, L)

        n_masked   = is_valid.float().sum(dim=1).clamp(min=1.0)
        ce_per_seq = (ce * is_valid.float()).sum(dim=1) / n_masked

        time_weights = 1.0 / (1.0 - timestep + 1e-4).to(ce_per_seq.device)
        time_weights = time_weights / time_weights.mean()

        loss = (ce_per_seq * time_weights).mean()

        return {
            "loss":            loss,
            "loss_unweighted": ce_per_seq.mean().item(),
            "n_masked":        is_valid.sum().item(),
        }
