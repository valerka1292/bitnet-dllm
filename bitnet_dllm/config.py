from __future__ import annotations
import json
from pathlib import Path
from pydantic import BaseModel, model_validator


_presets: dict[str, dict] = {
    "tiny":  dict(hidden_size=256,  num_layers=4,  num_heads=4,  ffn_hidden_size=683,  max_seq_len=256),
    "small": dict(hidden_size=768,  num_layers=12, num_heads=12, ffn_hidden_size=2048, max_seq_len=1024),
    "base":  dict(hidden_size=1024, num_layers=24, num_heads=16, ffn_hidden_size=2731, max_seq_len=2048),
    "large": dict(hidden_size=1536, num_layers=32, num_heads=24, ffn_hidden_size=4096, max_seq_len=4096),
}


def list_presets() -> list[str]:
    return list(_presets)


def register_preset(name: str, config: dict) -> None:
    _presets[name] = config


class BitDiffLMConfig(BaseModel):
    vocab_size:          int   = 32000
    hidden_size:         int   = 768
    num_layers:          int   = 12
    num_heads:           int   = 12
    ffn_hidden_size:     int   = 2048
    max_seq_len:         int   = 1024
    dropout:             float = 0.0
    activation_bits:     int   = 8

    mask_token_id:       int   = 103
    pad_token_id:        int   = 0
    bos_token_id:        int   = 1
    eos_token_id:        int   = 2

    noise_schedule:      str   = "log_linear"
    t_min:               float = 0.05
    t_max:               float = 0.99

    use_timestep_cond:   bool  = True
    timestep_freq_dim:   int   = 256

    min_lr_ratio:        float = 0.1
    warmup_ratio:        float = 0.05
    weight_decay:        float = 0.01
    time_eps:            float = 1e-4

    tie_word_embeddings: bool  = True

    @model_validator(mode="after")
    def _validate(self):
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size={self.hidden_size} not divisible by num_heads={self.num_heads}")
        if self.activation_bits not in (4, 8):
            raise ValueError(f"activation_bits must be 4 or 8, got {self.activation_bits}")
        if not (0 < self.t_min < self.t_max <= 1.0):
            raise ValueError(f"need 0 < t_min={self.t_min} < t_max={self.t_max} <= 1.0")
        for key in ("mask_token_id", "pad_token_id", "bos_token_id", "eos_token_id"):
            tid = getattr(self, key)
            if self.vocab_size <= tid:
                raise ValueError(f"vocab_size={self.vocab_size} must be > {key}={tid}")
        return self

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> BitDiffLMConfig:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        known = set(cls.model_fields)
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_preset(cls, name: str, **overrides) -> BitDiffLMConfig:
        if name not in _presets:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(_presets)}")
        return cls(**{**_presets[name], **overrides})
