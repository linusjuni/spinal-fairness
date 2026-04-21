from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import torch


PreprocessFn = Callable[[Path], torch.Tensor]


@dataclass(frozen=True)
class Encoder:
    """A loaded, frozen encoder with its companion preprocessing function.

    ``model(x)`` returns a ``(B, output_dim)`` tensor — pooling lives inside
    the model wrapper so the extraction loop stays encoder-agnostic.
    """

    model: torch.nn.Module
    preprocess: PreprocessFn
    output_dim: int


class EncoderFactory(Protocol):
    def __call__(self, device: str = "cuda") -> Encoder: ...
