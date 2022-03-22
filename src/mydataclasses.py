from torch import FloatTensor
from typing import Optional
from dataclasses import dataclass

from transformers.modeling_outputs import (
    Seq2SeqModelOutput,
)

@dataclass
class WithLossOutput(Seq2SeqModelOutput):
    loss: Optional[FloatTensor] = None

@dataclass
class WithLossAccOutput(Seq2SeqModelOutput):
    loss: Optional[FloatTensor] = None
    acc: Optional[FloatTensor] = None
    logits: Optional[FloatTensor] = None
