from __future__ import annotations
from typing import Callable
import torch
import torch.nn.functional as F

from .model import BitDiffLM


class MDLMAncestralSampler:
    def __init__(self, model: BitDiffLM, tokenizer, device: str = "cuda"):
        self.model         = model
        self.tokenizer     = tokenizer
        self.device        = device
        self.mask_id       = tokenizer.mask_token_id
        self.config        = model.config

    @torch.no_grad()
    def _denoise_loop(
        self,
        x_t:           torch.Tensor,
        frozen_mask:   torch.Tensor,
        num_steps:     int,
        temperature:   float,
        top_p:         float,
        step_callback: Callable | None,
    ) -> torch.Tensor:
        B, L = x_t.shape
        V    = self.config.vocab_size
        M    = self.mask_id
        model = self.model.eval()
        attn = torch.ones(B, L, device=self.device)
        t_seq = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)

        for i in range(num_steps):
            t_cur  = t_seq[i]
            t_next = t_seq[i + 1]
            t_b    = t_cur.expand(B)
            logits = model(x_t, attn, t_b)["logits"] / max(temperature, 1e-8)

            if top_p < 1.0:
                logits = self._top_p_filter(logits, top_p)

            probs     = F.softmax(logits, dim=-1)
            is_masked = (x_t == M) & (~frozen_mask)

            if i == num_steps - 1:
                x_t = torch.where(is_masked, probs.argmax(dim=-1), x_t)
            else:
                alpha_t      = 1.0 - t_cur
                alpha_t_next = 1.0 - t_next
                unmask_prob  = ((alpha_t_next - alpha_t) / (1.0 - alpha_t).clamp(min=1e-8)).clamp(0.0, 1.0)
                should_unmask = (torch.rand(B, L, device=self.device) < unmask_prob) & is_masked
                flat  = probs.view(B * L, V).clamp(min=1e-10)
                flat  = flat / flat.sum(-1, keepdim=True)
                samp  = torch.multinomial(flat, 1).view(B, L)
                x_t   = torch.where(should_unmask, samp, x_t)

            if step_callback is not None:
                decoded = self.decode(x_t)
                step_callback(i + 1, num_steps, t_cur, decoded)

        return x_t

    @torch.no_grad()
    def generate(
        self,
        prompt:        str | None = None,
        seq_len:       int        = 128,
        num_steps:     int        = 20,
        temperature:   float      = 1.0,
        top_p:         float      = 0.9,
        batch_size:    int        = 1,
        step_callback: Callable | None = None,
    ) -> list[str]:
        B, L = batch_size, seq_len
        M    = self.mask_id

        x_t    = torch.full((B, L), M, dtype=torch.long, device=self.device)
        frozen = torch.zeros(B, L, dtype=torch.bool, device=self.device)

        if prompt is not None:
            enc   = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=L)
            p_ids = enc.input_ids[0].to(self.device)
            p_len = min(p_ids.shape[0], L)
            x_t[:, :p_len]   = p_ids[:p_len]
            frozen[:, :p_len] = True

        x_final = self._denoise_loop(x_t, frozen, num_steps, temperature, top_p, step_callback)
        return self.decode(x_final)

    @torch.no_grad()
    def fill_mask(
        self,
        text:        str,
        num_steps:   int   = 10,
        temperature: float = 1.0,
        top_p:       float = 0.9,
    ) -> str:
        enc    = self.tokenizer(text, return_tensors="pt")
        ids    = enc.input_ids[0].to(self.device)
        L      = len(ids)
        x_t    = ids.unsqueeze(0)
        frozen = (x_t != self.mask_id)

        x_final = self._denoise_loop(x_t, frozen, num_steps, temperature, top_p, None)
        return self.decode(x_final)[0]

    def decode(self, ids: torch.Tensor) -> list[str]:
        return [self.tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in ids]

    def _top_p_filter(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        B, L, V = logits.shape
        flat    = logits.view(B * L, V)
        sorted_logits, sorted_idx = flat.sort(dim=-1, descending=True)
        probs_s  = sorted_logits.softmax(dim=-1)
        cum      = probs_s.cumsum(dim=-1)
        remove   = (cum - probs_s) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0]  = False
        remove   = remove.scatter(-1, sorted_idx, remove)
        return flat.masked_fill(remove, float("-inf")).view(B, L, V)
