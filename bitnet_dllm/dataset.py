from __future__ import annotations
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def worker_init_fn(worker_id: int):
    np.random.seed(torch.initial_seed() % (2 ** 32))


class MaskedDiffusionDataset(Dataset):
    def __init__(
        self,
        sequences:      list[torch.Tensor],
        mask_token_id:  int,
        pad_token_id:   int   = 0,
        noise_schedule: str   = "log_linear",
        t_min:          float = 0.05,
        t_max:          float = 0.99,
        min_seq_len:    int   = 4,
    ):
        self.mask_token_id  = mask_token_id
        self.pad_token_id   = pad_token_id
        self.noise_schedule = noise_schedule
        self.t_min          = t_min
        self.t_max          = t_max
        self.sequences      = [s for s in sequences if len(s) >= min_seq_len]
        if not self.sequences:
            raise ValueError(f"No sequences of length >= {min_seq_len}")

    @classmethod
    def from_texts(
        cls,
        texts:      list[str],
        tokenizer:  PreTrainedTokenizerBase,
        max_length: int = 512,
        **kwargs,
    ) -> MaskedDiffusionDataset:
        if tokenizer.mask_token_id is None:
            raise ValueError("Tokenizer has no [MASK] token.")
        seqs = []
        for text in texts:
            enc = tokenizer(text, max_length=max_length, truncation=True, padding=False, return_tensors="pt")
            seqs.append(enc.input_ids[0])
        return cls(seqs, tokenizer.mask_token_id, tokenizer.pad_token_id or 0, **kwargs)

    @classmethod
    def from_hf_dataset(
        cls,
        dataset,
        tokenizer:   PreTrainedTokenizerBase,
        text_column: str = "text",
        max_length:  int = 512,
        min_length:  int = 10,
        **kwargs,
    ) -> MaskedDiffusionDataset:
        texts = [t for t in dataset[text_column] if isinstance(t, str) and len(t.strip()) >= min_length]
        return cls.from_texts(texts, tokenizer, max_length=max_length, **kwargs)

    def _alpha(self, t: float) -> float:
        if self.noise_schedule == "log_linear":
            return 1.0 - t
        if self.noise_schedule == "cosine":
            return float(np.cos(t * np.pi / 2) ** 2)
        if self.noise_schedule == "sqrt":
            return float(1.0 - np.sqrt(t))
        raise ValueError(f"Unknown schedule: {self.noise_schedule}")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        ids       = self.sequences[idx].clone()
        t         = float(np.random.uniform(self.t_min, self.t_max))
        mask_prob = 1.0 - self._alpha(t)
        mask      = torch.rand(len(ids)) < mask_prob
        labels    = ids.clone()
        noisy     = ids.clone()
        noisy[mask] = self.mask_token_id
        return {
            "input_ids": noisy,
            "labels":    labels,
            "timestep":  torch.tensor(t, dtype=torch.float32),
        }

    def get_collate_fn(self):
        pad_id = self.pad_token_id
        def _collate(batch: list[dict]) -> dict:
            max_len = max(b["input_ids"].shape[0] for b in batch)
            B = len(batch)
            ids_buf = torch.full((B, max_len), pad_id, dtype=torch.long)
            lbl_buf = torch.full((B, max_len), -100, dtype=torch.long)
            msk_buf = torch.zeros((B, max_len), dtype=torch.long)
            t_buf   = torch.zeros(B, dtype=torch.float32)
            for i, b in enumerate(batch):
                n = b["input_ids"].shape[0]
                ids_buf[i, :n] = b["input_ids"]
                lbl_buf[i, :n] = b["labels"]
                msk_buf[i, :n] = 1
                t_buf[i]       = b["timestep"]
            return {
                "input_ids":      ids_buf,
                "labels":         lbl_buf,
                "attention_mask": msk_buf,
                "timestep":       t_buf,
            }
        return _collate
