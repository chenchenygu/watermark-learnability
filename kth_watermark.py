import random
from typing import Optional

import scipy.stats
import torch
from transformers import AutoTokenizer

DEFAULT_SEED = 42


class KTHWatermark:
    def __init__(
        self,
        vocab_size: int,
        key_len: int,
        seed: int = DEFAULT_SEED,
        device: Optional[str] = None,
        eps: float = 1e-20,
        random_shift: bool = False,
        num_shifts: Optional[int] = None,
    ):
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        generator = torch.Generator()  # generator is always cpu for reproducibility
        generator.manual_seed(seed)

        uniform = torch.clamp(torch.rand((key_len, vocab_size), generator=generator, dtype=torch.float32), min=eps)
        self.gumbel = (-torch.log(torch.clamp(-torch.log(uniform), min=eps))).to(device)

        if random_shift:
            if num_shifts is not None:
                self.possible_shifts = [i * (key_len // num_shifts) for i in range(num_shifts)]
            else:
                self.possible_shifts = list(range(key_len))

        self.random = random.Random(seed)  # for random shift
        self.seed = seed
        self.eps = eps
        self.vocab_size = vocab_size
        self.device = device
        self.key_len = key_len
        self.cur_shift = 0
        self.random_shift = random_shift
        self.num_shifts = num_shifts

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        index = (input_ids.shape[1] + self.cur_shift) % self.key_len
        gumbel = self.gumbel[index]  # (batch_size, vocab_size)
        return scores[..., :gumbel.shape[-1]] + gumbel
    
    def watermark_logits(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        logits: torch.FloatTensor,  # (batch, seq_len, vocab_size)
    ) -> torch.FloatTensor:
        """Returns watermarked logits to be used as distillation target."""
        index = torch.arange(input_ids.shape[1], device=input_ids.device) % self.key_len  # (seq_len,)
        gumbel = self.gumbel[index]  # (seq_len, vocab_size)
        # tokenizer vocab size and model outputs vocab size may be different
        logits[..., :gumbel.shape[-1]] += gumbel  # (batch, seq_len, vocab_size)
        return logits
        
    def watermark_logits_argmax(
        self,
        input_ids: torch.LongTensor,  # (batch, seq_len)
        logits: torch.FloatTensor,  # (batch, seq_len, vocab_size)
        random_shift: bool = False,
    ) -> torch.LongTensor:
        """Finds argmax token for watermark, returns token indexes to be used for cross-entropy loss.
        
        Returns tensor of shape (batch, seq_len), where each element is a token index.
        """
        shift = 0
        if self.random_shift:
            shift = self.random.choice(self.possible_shifts)
        index = (torch.arange(input_ids.shape[1], device=input_ids.device) + shift) % self.key_len  # (seq_len,)
        gumbel = self.gumbel[index]  # (seq_len, vocab_size)
        # tokenizer vocab size and model outputs vocab size may be different
        logits[..., :gumbel.shape[-1]] += gumbel  # (batch, seq_len, vocab_size)
        tokens = torch.argmax(logits, dim=-1)  # (batch, seq_len)
        return tokens
    
    # def detect(self, text: str, tokenizer) -> float:
    #     """
    #     Returns p-value, where null hypothesis is that the text is not watermarked.
        
    #     Under null hypothesis, each u is Uniform(0, 1), so each score (-log(1 - u)) is Exp(1).
    #     So the sum of scores is distributed as Gamma(n_tokens, 1).
    #     """
    #     tokens = tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)[0].to(self.device)  # (seq_len,)
    #     seq_len = tokens.shape[0]
    #     index = torch.arange(seq_len, device=tokens.device) % self.key_len  # (seq_len,)
    #     u = self.uniform[index, tokens]  # (seq_len,)
    #     score = torch.sum(-torch.log(1 - u + self.eps)).item()
    #     p_value = scipy.stats.gamma.sf(score, seq_len, loc=0, scale=1)
    #     return p_value


class KTHWatermarkDetector:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        key_len: int,
        seed: int = DEFAULT_SEED,
        eps: float = 1e-20,
    ):            
        generator = torch.Generator()  # generator is always cpu for reproducibility
        generator.manual_seed(seed)
        vocab_size = len(tokenizer)
        self.uniform = torch.clamp(
            torch.rand((key_len, vocab_size), generator=generator, dtype=torch.float32),
            min=eps,
            max=1 - eps,
        )
        self.seed = seed
        self.eps = eps
        self.vocab_size = vocab_size
        self.key_len = key_len

    def detect(self, text: str, shift: int = 0) -> float:
        """
        Returns p-value, where null hypothesis is that the text is not watermarked.
        
        Under null hypothesis, each u is Uniform(0, 1), so each score (-log(1 - u)) is Exp(1).
        So the sum of scores is distributed as Gamma(n_tokens, 1).
        """
        tokens = self.tokenizer.encode(text, return_tensors="pt", add_special_tokens=False)[0].to(self.device)  # (seq_len,)
        seq_len = tokens.shape[0]
        index = (torch.arange(seq_len, device=tokens.device) + shift) % self.key_len  # (seq_len,)
        u = self.uniform[index, tokens]  # (seq_len,)
        score = torch.sum(-torch.log(1 - u)).item()
        p_value = scipy.stats.gamma.sf(score, seq_len, loc=0, scale=1)
        return p_value
    