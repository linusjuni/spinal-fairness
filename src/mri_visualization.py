from __future__ import annotations

from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from nilearn import plotting

from src.utils.logger import get_logger

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
    dpi: int = 300,
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
        cut_coords = [_center_cut(img, display_mode)]

    logger.info(
        "Plotting slices", display_mode=display_mode, title=title or "(untitled)"
    )

    display = plotting.plot_anat(
        img,
        title=title,
        cut_coords=cut_coords,
        draw_cross=draw_cross,
        display_mode=display_mode,
    )

    if output_file:
        display.savefig(str(output_file), dpi=dpi)
        display.close()
        logger.info("Saved figure", path=str(output_file))
        return None

    return display


def plot_mri_with_seg(
    img: str | Path | nib.Nifti1Image,
    seg: str | Path | nib.Nifti1Image,
    *,
    title: str | None = None,
    output_file: str | Path | None = None,
    alpha: float = 0.35,
    dpi: int = 300,
) -> plt.Figure | None:
    """Plot the mid-sagittal slice with a two-label segmentation overlay.

    Reorients both volumes to RAS+ canonical so the sagittal through-plane
    is always axis-0, then displays the centre slice with:
      - label 1 (vertebral bodies) in red
      - label 2 (intervertebral discs) in green

    Parameters
    ----------
    img : path or nibabel image — the anatomical MRI background.
    seg : path or nibabel image — the segmentation mask (labels 1 and 2).
    title : optional display title.
    output_file : when set, saves the figure and returns None.
    alpha : opacity of the segmentation overlay (0–1).
    dpi : resolution when saving.

    Returns
    -------
    ``matplotlib.figure.Figure`` when *output_file* is None, otherwise None.
    """
    if not isinstance(img, nib.Nifti1Image):
        img = nib.load(str(img))
    if not isinstance(seg, nib.Nifti1Image):
        seg = nib.load(str(seg))

    img = nib.as_closest_canonical(img)
    seg = nib.as_closest_canonical(seg)

    mri_data = img.get_fdata()
    seg_data = seg.get_fdata().astype(np.uint8)

    # Axis-0 is left-right (sagittal through-plane) after canonical reorientation
    mid = mri_data.shape[0] // 2
    mri_slice = mri_data[mid]  # (AP, IS)
    seg_slice = seg_data[mid]  # (AP, IS)

    # Transpose so display rows = IS (vertical) and cols = AP (horizontal)
    mri_t = mri_slice.T
    seg_t = seg_slice.T

    zooms = img.header.get_zooms()
    aspect = float(zooms[1]) / float(zooms[2])  # AP_mm / IS_mm → square pixels

    nonzero = mri_data[mri_data > 0]
    vmin, vmax = np.percentile(nonzero, [1, 99]) if nonzero.size else (0.0, 1.0)

    fig, ax = plt.subplots(figsize=(5, 6))
    ax.imshow(
        mri_t,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        aspect=aspect,
        interpolation="bilinear",
    )

    overlay = np.zeros((*mri_t.shape, 4), dtype=np.float32)
    overlay[seg_t == 1] = [1.0, 0.15, 0.15, alpha]  # red   — vertebral body
    overlay[seg_t == 2] = [0.15, 0.9, 0.15, alpha]  # green — disc
    ax.imshow(overlay, origin="lower", aspect=aspect, interpolation="nearest")

    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)

    legend_patches = [
        mpatches.Patch(color=[1.0, 0.15, 0.15], label="Vertebral body"),
        mpatches.Patch(color=[0.15, 0.9, 0.15], label="Intervertebral disc"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.7)

    fig.tight_layout()
    logger.info("Plotting MRI+seg overlay", title=title or "(untitled)")

    if output_file:
        fig.savefig(str(output_file), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved figure", path=str(output_file))
        return None

    return fig
