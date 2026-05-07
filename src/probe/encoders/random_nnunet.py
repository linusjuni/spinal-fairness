"""Random-init nnU-Net null encoder for the demographic probe pipeline.

Same ResEncUNetL architecture and preprocessing as :mod:`nnunet`, but with no
trained weights — every parameter is drawn from a truncated Gaussian at
construction time. The purpose is to answer: *does the demographic signal in
the trained nnunet require learned segmentation features, or would any 3D
ResEncUNet pick it up from the nnunet-preprocessed inputs alone?*

Usage:
    uv run -m src.probe.pipeline random_nnunet
"""

from __future__ import annotations

import json

import torch
import torch.nn as nn

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from src.utils.logger import get_logger
from src.utils.settings import settings

from ._base import Encoder
from .nnunet import (
    CONFIG,
    DATASET_NAME,
    PLANS_NAME,
    NNUNetBottleneckEncoder,
    _build_preprocess,
    _case_id_mapping,
    _plans_path,
)

logger = get_logger(__name__)


def _init_weights(module: nn.Module) -> None:
    """trunc_normal_(std=0.02) for Conv3d/Linear, ones/zeros for InstanceNorm3d."""
    if isinstance(module, (nn.Linear, nn.Conv3d)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.InstanceNorm3d):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def load_random_nnunet(device: str = "cuda") -> Encoder:
    """Load random-init nnU-Net encoder (same arch + preprocess as nnunet, no checkpoint)."""
    plans_path = _plans_path()
    if not plans_path.exists():
        raise FileNotFoundError(f"Plans not found: {plans_path}")

    logger.info("Building random-init nnU-Net encoder", seed=settings.RANDOM_SEED)

    plans_manager = PlansManager(json.loads(plans_path.read_text()))
    config_manager = plans_manager.get_configuration(CONFIG)

    torch.manual_seed(settings.RANDOM_SEED)
    network = get_network_from_plans(
        config_manager.network_arch_class_name,
        config_manager.network_arch_init_kwargs,
        config_manager.network_arch_init_kwargs_req_import,
        input_channels=1,
        output_channels=3,
        allow_init=True,
        deep_supervision=False,
    )
    network.apply(_init_weights)

    output_dim = config_manager.network_arch_init_kwargs["features_per_stage"][-1]

    model = NNUNetBottleneckEncoder(network).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    b2nd_dir = (
        settings.nnUNet_preprocessed / DATASET_NAME / config_manager.data_identifier
    )
    patch_size = config_manager.patch_size
    filename_to_case_id = _case_id_mapping()
    preprocess = _build_preprocess(filename_to_case_id, b2nd_dir, patch_size)

    logger.success(
        "Loaded random-init nnU-Net encoder",
        output_dim=output_dim,
        device=device,
        seed=settings.RANDOM_SEED,
    )
    return Encoder(model=model, preprocess=preprocess, output_dim=output_dim)
