from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nilearn import plotting

from src.utils.logger import get_logger

if TYPE_CHECKING:
    import nibabel as nib

logger = get_logger("mri_visualization")


def plot_mri(
    img: str | Path | nib.Nifti1Image,
    *,
    display_mode: str = "x",
    title: str | None = None,
    output_file: str | Path | None = None,
    cut_coords: tuple[float, ...] | int | None = None,
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
    cut_coords : coordinates for the cuts, or None to auto-detect.
    draw_cross : whether to draw crosshairs on the slices.

    Returns
    -------
    nilearn display object when output_file is None, otherwise None.
    """
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
