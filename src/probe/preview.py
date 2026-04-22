"""Preview the MRI-CORE preprocessing pipelines.

Renders N exams as a 3-column grid:
    raw mid-sagittal | mri_core input | mri_core_cropped input

Sanity checks:
- Does `vol[X // 2]` actually land on the sagittal midline for this cohort?
- Does foreground_crop remove air padding without clipping tissue?
- Do the 1024x1024 network inputs look sensible?

Usage:
    uv run -m src.probe.preview          # default: 5 exams
    uv run -m src.probe.preview 10

Output: outputs/probe/preprocessing_preview.png
"""

from __future__ import annotations

import sys

import numpy as np

from src.data.exclusions import filter_excluded_cases
from src.data.loader import load_annotation_filenames
from src.data.schemas import Col
from src.probe.plots import preprocessing_preview_grid
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
    """What the encoder actually sees, in [0, 1] — stops short of ImageNet norm."""
    vol = load_ras_volume(nifti_path)
    slice_2d = mid_sagittal_slice(vol)
    if crop:
        slice_2d = foreground_crop(slice_2d)
    slice_2d = min_max_normalize(slice_2d)
    t = to_three_channel_tensor(slice_2d)
    t = resize_bilinear(t, INPUT_SIZE)
    return t[0].numpy()


def run(n_exams: int = 5) -> None:
    filenames = filter_excluded_cases(load_annotation_filenames(), logger)
    rng = np.random.default_rng(settings.RANDOM_SEED)
    picks = rng.choice(filenames.height, size=n_exams, replace=False)

    rows = []
    for pick in picks:
        row = filenames.row(int(pick), named=True)
        series_id = row[Col.SERIES_SUBMITTER_ID]
        path = settings.annotation_dir / row[Col.FILENAME]

        vol = load_ras_volume(path)
        raw = min_max_normalize(mid_sagittal_slice(vol))
        uncropped = _network_input(path, crop=False)
        cropped = _network_input(path, crop=True)

        rows.append({"series_id": series_id, "images": [raw, uncropped, cropped]})
        logger.info("Previewing", series=series_id)

    out_path = settings.OUTPUT_DIR / "probe" / "preprocessing_preview.png"
    preprocessing_preview_grid(
        rows,
        out_path=out_path,
        column_titles=[
            "raw mid-sagittal (display only)",
            "mri_core input (1024 x 1024)",
            "mri_core_cropped input (1024 x 1024)",
        ],
    )
    logger.success("Saved preview", path=str(out_path))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    run(n)
