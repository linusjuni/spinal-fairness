"""Random-init ViT-B null encoder.

Same ViT-B architecture and input pipeline as :mod:`mri_core`, but with no
pretrained weights — every parameter is drawn from a truncated Gaussian at
construction time. The purpose is to answer a narrower but cleaner question
than the cropped preprocessing ablation: *does the demographic signal in
MRI-CORE require learned features, or would any ViT pick it up from raw
pixel statistics?*

If probe AUROCs on these random features sit near chance, FOV and
intensity-histogram artefacts are ruled out — the signal requires learned
weights. If they stay high, the signal lives in raw image statistics
regardless of pretraining. See the demographic-probing ``findings.md``.

Initialization note: :class:`ImageEncoderViT` zero-initialises ``pos_embed``
(relying on the pretraining checkpoint to overwrite it), so an explicit
``trunc_normal_`` pass on ``pos_embed`` is load-bearing — otherwise the ViT
would be position-blind and the comparison with MRI-CORE would be unfair.
"""

from __future__ import annotations

import contextlib
import io
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.probe.vendored.sam import sam_model_registry
from src.utils.logger import get_logger
from src.utils.settings import settings

from ._base import Encoder
from .mri_core import INPUT_SIZE, OUTPUT_DIM, MRICoreEncoder, _preprocess

logger = get_logger(__name__)


def _init_weights(module: nn.Module) -> None:
    """timm-style ViT init: trunc_normal(std=0.02) for Linear/Conv/embeddings,
    zero biases, ones for LayerNorm weights."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _build_vit_random(seed: int) -> nn.Module:
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

    torch.manual_seed(seed)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sam = sam_model_registry["vit_b"](
            args,
            checkpoint=None,
            num_classes=args.num_cls,
            image_size=INPUT_SIZE,
            pretrained_sam=False,
        )
    vit = sam.image_encoder
    vit.apply(_init_weights)
    # ImageEncoderViT zero-inits pos_embed (image_encoder.py:76-77) — the SAM
    # checkpoint normally overwrites it, so for a random-init null we have to
    # fill it explicitly or the ViT becomes position-blind.
    if vit.pos_embed is not None:
        nn.init.trunc_normal_(vit.pos_embed, std=0.02)
    return vit


def load_random_vit_b(device: str = "cuda") -> Encoder:
    seed = settings.RANDOM_SEED
    logger.info("Building random-init ViT-B", seed=seed)
    vit = _build_vit_random(seed)
    model = MRICoreEncoder(vit).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.success(
        "Loaded random-init ViT-B",
        output_dim=OUTPUT_DIM,
        device=device,
        seed=seed,
    )
    return Encoder(model=model, preprocess=_preprocess, output_dim=OUTPUT_DIM)
