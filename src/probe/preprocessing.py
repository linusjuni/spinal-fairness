"""Shared preprocessing helpers used across encoders.

Each encoder module composes these into its own ``preprocess(path) -> Tensor``
function. Different encoders want different pipelines (2D slice vs. 3D volume,
different intensity schemes, different input sizes), so there is deliberately
no single project-wide preprocessing function.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_ras_volume(path: Path) -> np.ndarray:
    """Load a NIfTI volume and reorient to RAS+ canonical axes.

    Returns a float32 array shaped (X, Y, Z) where X=R (left->right),
    Y=A (posterior->anterior), Z=S (inferior->superior).
    """
    img = nib.as_closest_canonical(nib.load(path))
    return np.asarray(img.dataobj, dtype=np.float32)


def mid_sagittal_slice(vol: np.ndarray) -> np.ndarray:
    """Return the mid-sagittal 2D slice from a RAS volume.

    In RAS orientation, sagittal planes are indexed along axis 0 (R-L).
    Returns a (Y, Z) float32 slice.
    """
    return vol[vol.shape[0] // 2]


def min_max_normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Rescale an array linearly into [0, 1]."""
    lo, hi = float(arr.min()), float(arr.max())
    return (arr - lo) / max(hi - lo, eps)


def to_three_channel_tensor(slice_2d: np.ndarray) -> torch.Tensor:
    """(H, W) numpy -> (3, H, W) torch tensor, grayscale replicated to RGB."""
    t = torch.from_numpy(slice_2d).float().unsqueeze(0)
    return t.expand(3, -1, -1).contiguous()


def resize_bilinear(t: torch.Tensor, size: int) -> torch.Tensor:
    """Resize a (C, H, W) tensor to (C, size, size) via bilinear interpolation."""
    return F.interpolate(
        t.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def imagenet_normalize(t: torch.Tensor) -> torch.Tensor:
    """Subtract ImageNet mean and divide by std. Expects (3, H, W)."""
    return (t - _IMAGENET_MEAN) / _IMAGENET_STD


def foreground_crop(slice_2d: np.ndarray, threshold_frac: float = 0.05) -> np.ndarray:
    """Crop a 2D slice to the bounding box of pixels above the threshold.

    Default threshold is 5% of the slice's max intensity — above the typical
    Rician noise floor of MRI background, well below tissue intensities. If
    nothing clears the threshold (pathological / empty slice), the original
    slice is returned unchanged.

    Rationale: addresses the body-extent confound where fixed-FOV scans still
    contain variable amounts of air padding around the anatomy. Foreground-
    cropping + resize normalises the body-to-image ratio across scans.
    """
    thresh = float(slice_2d.max()) * threshold_frac
    mask = slice_2d > thresh
    if not mask.any():
        return slice_2d
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return slice_2d[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
