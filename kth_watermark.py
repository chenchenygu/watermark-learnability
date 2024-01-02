import random
from typing import Optional

import torch

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
    