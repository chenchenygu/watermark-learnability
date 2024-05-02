from typing import Optional

import scipy.stats
import torch
from transformers import AutoTokenizer

from watermarks.watermark_types import WatermarkType

DEFAULT_SEED = 42


class AarWatermark:
    def __init__(
        self,
        vocab_size: int,
        k: int,
        seed: int = DEFAULT_SEED,
        eps: float = 1e-20,
        device: Optional[str] = None,
    ):
        self.type = WatermarkType.AAR
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        generator = torch.Generator()  # generator is always cpu for reproducibility
        generator.manual_seed(seed)

        # clamp to avoid NaNs
        uniform = torch.clamp(torch.rand((vocab_size * k, vocab_size), generator=generator, dtype=torch.float32), min=eps)
        self.gumbel = (-torch.log(torch.clamp(-torch.log(uniform), min=eps))).to(device)

        self.k = k
        self.vocab_size = vocab_size
        self.seed = seed
        self.eps = eps
        self.device = device

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if input_ids.shape[-1] < self.k:
            return scores
        prev_token = torch.sum(input_ids[:, -self.k:], dim=-1)  # (batch_size,)
        gumbel = self.gumbel[prev_token]  # (batch_size, vocab_size)
        return scores[..., :gumbel.shape[-1]] + gumbel
        
    def watermark_logits_argmax(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        logits: torch.FloatTensor,  # (batch, seq_len, vocab_size)
    ) -> torch.LongTensor:
        """Finds argmax token for watermark, returns token indexes to be used for cross-entropy loss.
        
        Returns tensor of shape (batch, seq_len), where each element is a token index.
        """
        hashes = torch.sum(input_ids.unfold(-1, self.k, 1), dim=-1)  # (batch, seq_len - k + 1)
        gumbel = self.gumbel[hashes]  # (batch, seq_len - k + 1, vocab_size)
        # tokenizer vocab size and model outputs vocab size may be different
        logits[..., self.k - 1:, :gumbel.shape[-1]] += gumbel  # (batch, seq_len, vocab_size)
        tokens = torch.argmax(logits, dim=-1)  # (batch, seq_len)
        return tokens
    

class AarWatermarkDetector:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        k: int = 1,
        seed: int = DEFAULT_SEED,
        eps: float = 1e-20,
    ):
        generator = torch.Generator()  # generator is always cpu for reproducibility
        generator.manual_seed(seed)
        vocab_size = len(tokenizer)
        self.uniform = torch.clamp(
            torch.rand((vocab_size * k, vocab_size), generator=generator, dtype=torch.float32),
            min=eps,
            max=1 - eps,
        )

        self.tokenizer = tokenizer
        self.k = k
        self.seed = seed
        self.eps = eps
        self.vocab_size = vocab_size
        
    def detect(self, text: str) -> float:
        """
        Returns p-value, where null hypothesis is that the text is not watermarked.
        
        Under null hypothesis, each u is Uniform(0, 1), so each score (-log(1 -u )) is Exp(1).
        So the sum of scores is distributed as Gamma(n_tokens, 1).
        """
        tokens = self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)[0]  # (seq_len,)
        seq_len = tokens.shape[0]
        score = 0
        for i in range(self.k, seq_len):
            prev_tokens_sum = torch.sum(tokens[i - self.k:i], dim=-1)
            token = tokens[i]
            u = self.uniform[prev_tokens_sum, token]
            score += -torch.log(1 - u)
        p_value = scipy.stats.gamma.sf(score, seq_len - self.k, loc=0, scale=1)
        return p_value
