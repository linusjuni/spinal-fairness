"""Preview preprocessing pipelines for the MRI-CORE encoder variants.

Renders N random exams as a 3-column grid:
    raw mid-sagittal | mri_core input | mri_core_cropped input

Sanity checks:
- Does `vol[X // 2]` actually land on the sagittal midline for this cohort?
- Does foreground_crop remove air padding without clipping tissue?
- Do the 1024x1024 network inputs look sensible?

Usage:
    uv run -m src.probe.preview_preprocessing          # default: 5 exams
    uv run -m src.probe.preview_preprocessing 10       # custom count

Output: outputs/probe/preprocessing_preview.png
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np

from src.data.exclusions import filter_excluded_cases
from src.data.loader import load_annotation_filenames
from src.data.schemas import Col
from src.probe.preprocessing import (
    foreground_crop,
    load_ras_volume,
    mid_sagittal_slice,
    min_max_normalize,
    resize_bilinear,
    to_three_channel_tensor,
)
from src.utils.logger import get_logger
from src.utils.settings import settings

logger = get_logger(__name__)

INPUT_SIZE = 1024


def _network_input(nifti_path, *, crop: bool) -> np.ndarray:
    """The (grayscale) image the encoder actually sees, in [0, 1].

    Stops short of ImageNet normalisation — those values are negative and
    hard to eyeball. Everything else (slice selection, optional crop,
    min-max, replicate, resize) matches the production pipeline.
    """
    vol = load_ras_volume(nifti_path)
    slice_2d = mid_sagittal_slice(vol)
    if crop:
        slice_2d = foreground_crop(slice_2d)
    slice_2d = min_max_normalize(slice_2d)
    t = to_three_channel_tensor(slice_2d)
    t = resize_bilinear(t, INPUT_SIZE)
    return t[0].numpy()


def _for_display(slice_2d: np.ndarray) -> np.ndarray:
    """Rotate 90 degrees CCW so the superior-inferior axis runs vertically."""
    return np.rot90(slice_2d, k=1)


def run(n_exams: int = 5) -> None:
    filenames = filter_excluded_cases(load_annotation_filenames(), logger)
    rng = np.random.default_rng(settings.RANDOM_SEED)
    picks = rng.choice(filenames.height, size=n_exams, replace=False)

    fig, axes = plt.subplots(
        n_exams, 3, figsize=(12, 4 * n_exams), squeeze=False
    )
    axes[0, 0].set_title("raw mid-sagittal (display only)")
    axes[0, 1].set_title("mri_core input (1024 x 1024)")
    axes[0, 2].set_title("mri_core_cropped input (1024 x 1024)")

    for row_idx, pick in enumerate(picks):
        row = filenames.row(int(pick), named=True)
        series_id = row[Col.SERIES_SUBMITTER_ID]
        path = settings.annotation_dir / row[Col.FILENAME]

        vol = load_ras_volume(path)
        raw = mid_sagittal_slice(vol)
        raw = min_max_normalize(raw)  # rescale for display only
        uncropped = _network_input(path, crop=False)
        cropped = _network_input(path, crop=True)

        for col_idx, slice_ in enumerate([raw, uncropped, cropped]):
            ax = axes[row_idx, col_idx]
            ax.imshow(_for_display(slice_), cmap="gray")
            ax.axis("off")

        axes[row_idx, 0].text(
            0.02,
            0.98,
            series_id,
            transform=axes[row_idx, 0].transAxes,
            color="yellow",
            fontsize=8,
            va="top",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 2},
        )
        logger.info("Previewing", series=series_id)

    out_dir = settings.OUTPUT_DIR / "probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preprocessing_preview.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.success("Saved preview", path=str(out_path))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run(n)
