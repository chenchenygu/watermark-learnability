from typing import List, Optional

import torch
from transformers import AutoTokenizer

from watermarks.kgw.watermark_processor import WatermarkBase
from watermarks.watermark_types import WatermarkType


class KGWWatermark:
    def __init__(
        self,
        vocab: List[int] = None,
        gamma: float = 0.5,
        delta: float = 2.0,
        seeding_scheme: str = "simple_1",
        tokenizer: AutoTokenizer = None,
        device: Optional[str] = None,
    ):
        self.type = WatermarkType.KGW
        self.watermark_base = WatermarkBase(
            vocab=vocab,
            gamma=gamma,
            delta=delta,
            seeding_scheme=seeding_scheme,
            device="cpu",  # cpu for reproducibility
        )
        self.kgw_device = "cpu"
        self.k = self.watermark_base.context_width
        self.greenlist_masks = torch.full(
            (self.k * self.watermark_base.vocab_size, self.watermark_base.vocab_size),
            fill_value=False,
            dtype=bool,
        )
        for i in range(self.greenlist_masks.shape[0]):
            greenlist_ids = self.watermark_base._get_greenlist_ids(torch.tensor([0] * (self.k - 1) + [i], dtype=torch.long))
            self.greenlist_masks[i, greenlist_ids] = True

        self.greenlist_masks = self.greenlist_masks.to(device)

        # save watermark base parameters
        self.vocab = self.watermark_base.vocab
        self.vocab_size = self.watermark_base.vocab_size
        self.gamma = self.watermark_base.gamma
        self.delta = self.watermark_base.delta
        self.seeding_scheme = self.watermark_base.seeding_scheme
        self.hash_key = self.watermark_base.hash_key
        self.select_green_tokens = self.watermark_base.select_green_tokens
        
        if tokenizer is not None and seeding_scheme == "simple_1":
            # remove special tokens from greenlists
            if tokenizer.eos_token_id is not None:
                self.greenlist_masks[:, tokenizer.eos_token_id] = False
                self.greenlist_masks[tokenizer.eos_token_id, :] = False
            if tokenizer.bos_token_id is not None:
                self.greenlist_masks[:, tokenizer.bos_token_id] = False
                self.greenlist_masks[tokenizer.bos_token_id, :] = False
            if tokenizer.pad_token_id is not None:
                self.greenlist_masks[:, tokenizer.pad_token_id] = False
                self.greenlist_masks[tokenizer.pad_token_id, :] = False
            if tokenizer.unk_token_id is not None:
                self.greenlist_masks[:, tokenizer.unk_token_id] = False
                self.greenlist_masks[tokenizer.unk_token_id, :] = False

    def watermark_logits(self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        logits: torch.FloatTensor,  # (batch, seq_len, vocab_size)
    ) -> torch.FloatTensor:
        """Returns watermarked logits to be used as distillation target."""
        hashes = torch.sum(input_ids.unfold(-1, self.k, 1), dim=-1)  # (batch, seq_len - k + 1)
        mask = self.greenlist_masks[hashes]  # (batch, seq_len - k + 1, vocab_size)
        # tokenizer vocab size and model outputs vocab size may be different
        logits[..., self.k - 1:, :mask.shape[-1]][mask] += self.delta
        return logits
