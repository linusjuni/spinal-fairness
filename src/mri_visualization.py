from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import nibabel as nib
import numpy as np
from nilearn import plotting

from src.utils.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger("mri_visualization")

_SINGLE_AXIS_MODES = {"x": 0, "y": 1, "z": 2}


def _center_cut(img: str | Path | nib.Nifti1Image, display_mode: str) -> float:
    """World-space coordinate of the image center for single-axis display modes."""
    if isinstance(img, (str, Path)):
        img = nib.load(str(img))
    axis = _SINGLE_AXIS_MODES[display_mode]
    center_vox = np.array(img.shape[:3], dtype=float) / 2
    center_world = img.affine @ np.append(center_vox, 1.0)
    return float(center_world[axis])


def plot_mri(
    img: str | Path | nib.Nifti1Image,
    *,
    display_mode: str = "x",
    title: str | None = None,
    output_file: str | Path | None = None,
    cut_coords: tuple[float, ...] | int | float | None = None,
    draw_cross: bool = False,
):
    """Plot anatomical slices of an MRI volume.

    Parameters
    ----------
    img : path or nibabel image
        NIfTI file to visualize (nilearn accepts both).
    display_mode : nilearn display mode (default ``"x"`` for sagittal).
    title : optional display title
    output_file : when set, saves the figure to this path and returns None.
    cut_coords : coordinates for the cuts. For single-axis modes (``"x"``,
        ``"y"``, ``"z"``), defaults to the image center so the slice is never
        black. Pass an int to request that many auto-spaced cuts instead.
    draw_cross : whether to draw crosshairs on the slices.

    Returns
    -------
    nilearn display object when output_file is None, otherwise None.
    """
    if cut_coords is None and display_mode in _SINGLE_AXIS_MODES:
        cut_coords = _center_cut(img, display_mode)

    logger.info("Plotting slices", display_mode=display_mode, title=title or "(untitled)")

    display = plotting.plot_anat(
        img,
        title=title,
        output_file=str(output_file) if output_file else None,
        cut_coords=cut_coords,
        draw_cross=draw_cross,
        display_mode=display_mode,
    )

    if output_file:
        logger.info("Saved figure", path=str(output_file))

    return display
