from __future__ import annotations
import os
from pathlib import Path
from pydantic import BaseModel, field_validator


class TokenizerConfig(BaseModel):
    vocab_size:    int
    mask_token_id: int | None
    pad_token_id:  int | None
    bos_token_id:  int | None
    eos_token_id:  int | None

    @field_validator("mask_token_id")
    @classmethod
    def _mask_required(cls, v):
        if v is None:
            raise ValueError("Tokenizer must have a [MASK] token for diffusion training")
        return v


class BitDiffTokenizer:
    """
    Tokenizer utilities for BitDiffLM.

    Supports:
    - Training a BPE tokenizer from scratch
    - Loading a pre-trained HuggingFace tokenizer
    - Saving/loading to/from disk
    """

    SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    @classmethod
    def train(
        cls,
        texts_or_files:   list[str],
        vocab_size:       int  = 30000,
        min_frequency:    int  = 2,
        save_dir:         str | Path | None = None,
        from_files:       bool = False,
    ):
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
        from transformers import PreTrainedTokenizerFast

        tok = Tokenizer(models.BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder       = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=cls.SPECIAL_TOKENS,
        )

        if from_files:
            tok.train(texts_or_files, trainer)
        else:
            tok.train_from_iterator(texts_or_files, trainer)

        cls_id = tok.token_to_id("[CLS]")
        sep_id = tok.token_to_id("[SEP]")
        tok.post_processor = processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
        )

        hf_tok = PreTrainedTokenizerFast(
            tokenizer_object=tok,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )

        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            hf_tok.save_pretrained(save_dir)

        return hf_tok

    @classmethod
    def load(cls, path: str | Path):
        from transformers import PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast.from_pretrained(str(path))
        cls._check(tok, path)
        return tok

    @classmethod
    def from_pretrained(cls, name_or_path: str):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(name_or_path)
        cls._check(tok, name_or_path)
        return tok

    @staticmethod
    def _check(tok, src):
        if tok.mask_token_id is None:
            raise ValueError(
                f"Tokenizer '{src}' has no [MASK] token. "
                "Use a BERT-compatible tokenizer or train one with BitDiffTokenizer.train()."
            )

    @staticmethod
    def config_from_tokenizer(tokenizer, config):
        """Sync config special token IDs from tokenizer with schema validation."""
        data = dict(
            vocab_size    = len(tokenizer),
            mask_token_id = tokenizer.mask_token_id,
            pad_token_id  = tokenizer.pad_token_id or 0,
            bos_token_id  = tokenizer.cls_token_id,
            eos_token_id  = tokenizer.sep_token_id,
        )
        TokenizerConfig(**data)  # validate before mutating config
        config.vocab_size    = data["vocab_size"]
        config.mask_token_id = data["mask_token_id"]
        config.pad_token_id  = data["pad_token_id"]
        if data["bos_token_id"] is not None:
            config.bos_token_id = data["bos_token_id"]
        if data["eos_token_id"] is not None:
            config.eos_token_id = data["eos_token_id"]
        return config
