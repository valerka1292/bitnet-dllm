from __future__ import annotations
import torch
import numpy as np
from torch.utils.data import Dataset


def worker_init_fn(worker_id: int):
    np.random.seed(torch.initial_seed() % (2 ** 32))


class MaskedDiffusionDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        mask_token_id:  int,
        pad_token_id:   int   = 0,
        noise_schedule: str   = "log_linear",
        t_min:          float = 0.05,
        t_max:          float = 0.99,
    ):
        if mask_token_id < 0 or mask_token_id == pad_token_id:
            raise ValueError(f"mask_token_id={mask_token_id} must be >= 0 and != pad_token_id={pad_token_id}")
        if not (0.0 < t_min < t_max <= 1.0):
            raise ValueError(f"need 0 < t_min={t_min} < t_max={t_max} <= 1.0")
        self.hf_dataset      = hf_dataset
        self.mask_token_id   = mask_token_id
        self.pad_token_id    = pad_token_id
        self.noise_schedule  = noise_schedule
        self.t_min           = t_min
        self.t_max           = t_max
        self._input_ids_key  = "input_ids"

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        if self.noise_schedule == "log_linear":
            return 1.0 - t
        if self.noise_schedule == "cosine":
            return (t * (torch.pi / 2)).cos() ** 2
        if self.noise_schedule == "sqrt":
            return 1.0 - t.sqrt()
        raise ValueError(f"Unknown schedule: {self.noise_schedule}")

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.hf_dataset[idx]
        if isinstance(item, dict):
            ids = item[self._input_ids_key]
        elif isinstance(item, torch.Tensor):
            ids = item
        else:
            ids = item
        return {"input_ids": ids}

    def get_collate_fn(self):
        pad_id = self.pad_token_id
        t_min = self.t_min
        t_max = self.t_max
        alpha_fn = self._alpha
        mask_id = self.mask_token_id

        def _collate(batch: list[dict]) -> dict:
            B = len(batch)
            ids_list = [torch.as_tensor(b["input_ids"], dtype=torch.long) for b in batch]
            lengths = [ids.shape[0] for ids in ids_list]
            max_len = max(lengths)

            padded = torch.full((B, max_len), pad_id, dtype=torch.long)
            labels = torch.full((B, max_len), -100, dtype=torch.long)
            attn_mask = torch.zeros((B, max_len), dtype=torch.long)

            for i, ids in enumerate(ids_list):
                n = lengths[i]
                padded[i, :n] = ids
                labels[i, :n] = ids
                attn_mask[i, :n] = 1

            t = torch.empty(B, dtype=torch.float32).uniform_(t_min, t_max)
            mask_prob = 1.0 - alpha_fn(t)
            mask = torch.rand(B, max_len) < mask_prob[:, None]
            noisy = padded.clone()
            noisy[mask] = mask_id

            return {
                "input_ids":      noisy,
                "labels":         labels,
                "attention_mask": attn_mask,
                "timestep":       t,
            }

        return _collate
