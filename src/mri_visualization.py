from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nilearn import plotting

from src.utils.logger import get_logger

if TYPE_CHECKING:
    import nibabel as nib

logger = get_logger("mri_visualization")


def plot_ortho_slices(
    img: str | Path | nib.Nifti1Image,
    *,
    title: str | None = None,
    output_file: str | Path | None = None,
    cut_coords: tuple[float, float, float] | None = None,
    draw_cross: bool = False,
):
    """Plot orthogonal anatomical slices of an MRI volume.

    Parameters
    ----------
    img : path or nibabel image
        NIfTI file to visualize (nilearn accepts both).
    title : optional display title
    output_file : when set, saves the figure to this path and returns None.
    cut_coords : (x, y, z) coordinates for the cut, or None to auto-detect center.
    draw_cross : whether to draw crosshairs on the slices.

    Returns
    -------
    nilearn display object when output_file is None, otherwise None.
    """
    logger.info("Plotting ortho slices", title=title or "(untitled)")

    display = plotting.plot_anat(
        img,
        title=title,
        output_file=str(output_file) if output_file else None,
        cut_coords=cut_coords,
        draw_cross=draw_cross,
        display_mode="ortho",
    )

    if output_file:
        logger.info("Saved ortho slices", path=str(output_file))

    return display
