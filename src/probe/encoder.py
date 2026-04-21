from __future__ import annotations

from pathlib import Path

import torch

from src.utils.settings import settings

MEDICALNET_WEIGHTS = settings.MODELS_DIR / "medicalnet" / "resnet_50_23dataset.pth"


def load_medicalnet(weights_path: Path = MEDICALNET_WEIGHTS) -> torch.nn.Module:
    """Load MedicalNet ResNet-50 pretrained on the 23-dataset mix.

    The published checkpoint is a segmentation model saved under DataParallel:
      - Top-level key is "state_dict".
      - Every parameter key is prefixed with "module.".
      - Includes a "conv_seg" head we strip, since we use the backbone as a
        frozen feature extractor (layer4 → adaptive_avg_pool3d → 2048-d).
    """
    # TODO: import the 3D ResNet-50 architecture. Options:
    #   (a) clone Tencent/MedicalNet and import from models/resnet.py
    #   (b) use monai.networks.nets.resnet50 with spatial_dims=3, which is
    #       key-compatible with MedicalNet weights
    # model = ...

    ckpt = torch.load(weights_path, map_location="cpu")
    state_dict = {
        k.replace("module.", ""): v
        for k, v in ckpt["state_dict"].items()
        if not k.startswith("module.conv_seg")
    }
    # model.load_state_dict(state_dict, strict=False)
    # model.eval()
    # return model
    raise NotImplementedError("Wire up the 3D ResNet-50 architecture above")


# Run a single NIfTI volume through the frozen encoder and return a 1D
# feature vector (2048-d after adaptive_avg_pool3d on layer4).
# def embed_volume(model: torch.nn.Module, nifti_path: Path) -> np.ndarray: ...
