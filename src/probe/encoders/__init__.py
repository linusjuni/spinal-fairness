from __future__ import annotations

from ._base import Encoder, EncoderFactory
from .mri_core import load_mri_core, load_mri_core_cropped
from .random_vit_b import load_random_vit_b


REGISTRY: dict[str, EncoderFactory] = {
    "mri_core": load_mri_core,
    "mri_core_cropped": load_mri_core_cropped,
    "random_vit_b": load_random_vit_b,
}


def load_encoder(name: str, device: str = "cuda") -> Encoder:
    """Load a frozen encoder by its registry name."""
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown encoder {name!r}. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](device=device)


__all__ = ["Encoder", "EncoderFactory", "REGISTRY", "load_encoder"]
