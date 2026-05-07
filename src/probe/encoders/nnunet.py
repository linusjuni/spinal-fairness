"""nnU-Net bottleneck encoder for the demographic probe pipeline.

Loads a trained nnU-Net (ResEncUNet, 3d_fullres, fold 0) and extracts
global-average-pooled bottleneck features. Preprocessing loads the
already-preprocessed .npz files from $nnUNet_preprocessed so the features
match exactly what the model saw during training.

Usage:
    uv run -m src.probe.pipeline nnunet
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

from src.utils.logger import get_logger
from src.utils.settings import settings

from ._base import Encoder

logger = get_logger(__name__)

DATASET_NAME = "Dataset001_CSpineSeg"
CONFIG = "3d_fullres"
FOLD = 0
PLANS_NAME = "nnUNetResEncUNetLPlans"
TRAINER_NAME = "nnUNetTrainerWandB"


def _plans_path() -> Path:
    return settings.nnUNet_preprocessed / DATASET_NAME / f"{PLANS_NAME}.json"


def _checkpoint_path() -> Path:
    return (
        settings.nnUNet_results
        / DATASET_NAME
        / f"{TRAINER_NAME}__{PLANS_NAME}__{CONFIG}"
        / f"fold_{FOLD}"
        / "checkpoint_final.pth"
    )


def _preprocessed_dir() -> Path:
    return settings.nnUNet_preprocessed / DATASET_NAME / f"{PLANS_NAME}_{CONFIG}"


def _case_id_mapping() -> dict[str, str]:
    """Return filename -> case_id mapping from Dataset001's case_id_mapping.json."""
    mapping_path = settings.nnUNet_raw / DATASET_NAME / "case_id_mapping.json"
    records = json.loads(mapping_path.read_text())
    return {r["source_filename"]: r["case_id"] for r in records}


class NNUNetBottleneckEncoder(nn.Module):
    """Wraps nnU-Net encoder to return GAP'd bottleneck features."""

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.encoder = network.encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = self.encoder(x)
        bottleneck = skips[-1]
        return bottleneck.mean(dim=tuple(range(2, bottleneck.ndim)))


def _build_preprocess(filename_to_case_id: dict[str, str]) -> callable:
    """Build a preprocess function that loads .npz files by NIfTI filename."""
    npz_dir = _preprocessed_dir()

    def preprocess(nifti_path: Path) -> torch.Tensor:
        filename = nifti_path.name
        case_id = filename_to_case_id.get(filename)
        if case_id is None:
            raise FileNotFoundError(f"No case_id mapping for {filename}")

        npz_path = npz_dir / f"{case_id}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Preprocessed .npz not found: {npz_path}")

        data = np.load(npz_path)["data"]
        return torch.from_numpy(data).float()

    return preprocess


def load_nnunet(device: str = "cuda") -> Encoder:
    """Load frozen nnU-Net encoder (3d_fullres, fold 0, Dataset001)."""
    plans_path = _plans_path()
    checkpoint_path = _checkpoint_path()

    if not plans_path.exists():
        raise FileNotFoundError(f"Plans not found: {plans_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(
        "Loading nnU-Net encoder",
        config=CONFIG,
        fold=FOLD,
        checkpoint=checkpoint_path.name,
    )

    plans_manager = PlansManager(json.loads(plans_path.read_text()))
    config_manager = plans_manager.get_configuration(CONFIG)

    network = get_network_from_plans(
        config_manager.network_arch_class_name,
        config_manager.network_arch_init_kwargs,
        config_manager.network_arch_init_kwargs_req_import,
        input_channels=1,
        output_channels=3,
        allow_init=False,
        deep_supervision=False,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = {}
    for k, v in checkpoint["network_weights"].items():
        key = k[7:] if k.startswith("module.") else k
        state_dict[key] = v
    network.load_state_dict(state_dict, strict=False)

    output_dim = config_manager.network_arch_init_kwargs["features_per_stage"][-1]

    model = NNUNetBottleneckEncoder(network).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    filename_to_case_id = _case_id_mapping()
    preprocess = _build_preprocess(filename_to_case_id)

    logger.success("Loaded nnU-Net encoder", output_dim=output_dim, device=device)
    return Encoder(model=model, preprocess=preprocess, output_dim=output_dim)
