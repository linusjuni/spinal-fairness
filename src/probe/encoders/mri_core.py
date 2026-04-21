"""MRI-CORE ViT-B frozen feature extractor.

Loads the released ``mri_foundation.pth`` checkpoint into the vendored SAM
image encoder, bypasses the (randomly-initialised) neck, and mean-pools
the last transformer block's output to a 768-d vector.

See docs/demographic-probing-of-medical-image-encoders/methodology.md and
encoder-recommendations.md for the rationale behind the preprocessing
choices (per-slice min-max + ImageNet norm matches pretraining).
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.probe.preprocessing import (
    imagenet_normalize,
    load_ras_volume,
    mid_sagittal_slice,
    min_max_normalize,
    resize_bilinear,
    to_three_channel_tensor,
)
from src.probe.vendored.sam import sam_model_registry
from src.utils.logger import get_logger
from src.utils.settings import settings

from ._base import Encoder

logger = get_logger(__name__)


MRI_CORE_WEIGHTS = settings.MODELS_DIR / "mri_core" / "MRI_CORE_vitb.pth"
INPUT_SIZE = 1024
OUTPUT_DIM = 768


class MRICoreEncoder(nn.Module):
    """Wraps the vendored ImageEncoderViT to return mean-pooled, pre-neck features.

    The released ``teacher`` state dict contains DINOv2-trained image-encoder
    blocks but not the SAM neck — the neck is silently random-initialised on
    load. We therefore take the last block's output (shape ``(B, H/16, W/16, 768)``
    at 1024 input) and spatial-mean-pool it, matching the linear-probing recipe
    reported in the MRI-CORE paper.
    """

    def __init__(self, vit: nn.Module) -> None:
        super().__init__()
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.vit.patch_embed(x)
        if self.vit.pos_embed is not None:
            h = h + self.vit.pos_embed
        for blk in self.vit.blocks:
            h = blk(h)
        return h.mean(dim=(1, 2))


def _build_vit(checkpoint: Path) -> nn.Module:
    # The repo's _build_sam reads a handful of attrs off an argparse Namespace.
    # Build a minimal SimpleNamespace rather than calling cfg.parse_args() —
    # the latter would consume sys.argv.
    args = SimpleNamespace(
        arch="vit_b",
        if_encoder_adapter=False,
        encoder_adapter_depths=[],
        if_mask_decoder_adapter=False,
        decoder_adapt_depth=0,
        if_encoder_lora_layer=False,
        if_decoder_lora_layer=False,
        encoder_lora_layer=[],
        num_cls=1,
        image_size=INPUT_SIZE,
    )

    # The builder prints a long key-renaming trace; swallow it.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sam = sam_model_registry["vit_b"](
            args,
            checkpoint=str(checkpoint),
            num_classes=args.num_cls,
            image_size=INPUT_SIZE,
            pretrained_sam=False,
        )
    return sam.image_encoder


def _preprocess(nifti_path: Path) -> torch.Tensor:
    """NIfTI -> (3, 1024, 1024) float tensor, ImageNet-normalised."""
    vol = load_ras_volume(nifti_path)
    slice_2d = mid_sagittal_slice(vol)
    slice_2d = min_max_normalize(slice_2d)
    t = to_three_channel_tensor(slice_2d)
    t = resize_bilinear(t, INPUT_SIZE)
    return imagenet_normalize(t)


def load_mri_core(device: str = "cuda") -> Encoder:
    if not MRI_CORE_WEIGHTS.exists():
        raise FileNotFoundError(
            f"MRI-CORE weights not found at {MRI_CORE_WEIGHTS}. Download "
            f"mri_foundation.pth from the repo README's Google Drive link "
            f"(https://github.com/mazurowski-lab/mri_foundation) and save it "
            f"as {MRI_CORE_WEIGHTS.name} under {MRI_CORE_WEIGHTS.parent}/."
        )
    logger.info("Loading MRI-CORE", weights=MRI_CORE_WEIGHTS.name)
    vit = _build_vit(MRI_CORE_WEIGHTS)
    model = MRICoreEncoder(vit).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.success("Loaded MRI-CORE", output_dim=OUTPUT_DIM, device=device)
    return Encoder(model=model, preprocess=_preprocess, output_dim=OUTPUT_DIM)
