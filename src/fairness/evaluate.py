"""Evaluate NIfTI predictions against reference labels.

Computes per-case Dice (and optionally HD95) for each segmentation label,
emitting a CSV with one row per case. This is the only module in src/fairness
that touches NIfTI I/O.

Usage:
    uv run -m src.fairness.evaluate \
        --predictions <dir> --references <dir> \
        --mapping <case_id_mapping.json> --output <csv> \
        [--split test] [--metrics dice hd95]
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import polars as pl

from src.fairness import LABELS
from src.utils.logger import get_logger

logger = get_logger("fairness.evaluate")

VALID_METRICS = {"dice", "hd95"}


def dice_coefficient(pred: np.ndarray, ref: np.ndarray, label: int) -> float:
    """Binary Dice coefficient for a single label.

    Returns NaN if both masks are empty (nnU-Net convention).
    Returns 0.0 if one mask is empty and the other is not.
    """
    pred_mask = pred == label
    ref_mask = ref == label

    pred_sum = pred_mask.sum()
    ref_sum = ref_mask.sum()

    if pred_sum == 0 and ref_sum == 0:
        return float("nan")
    if pred_sum == 0 or ref_sum == 0:
        return 0.0

    intersection = (pred_mask & ref_mask).sum()
    return float(2.0 * intersection / (pred_sum + ref_sum))


def hausdorff_95(
    pred: np.ndarray,
    ref: np.ndarray,
    label: int,
    spacing: tuple[float, ...],
) -> float:
    """95th-percentile Hausdorff distance in mm for a single label.

    Returns NaN if both masks are empty.
    Returns inf if exactly one mask is empty.
    """
    from surface_distance import (
        compute_robust_hausdorff,
        compute_surface_distances,
    )

    pred_mask = pred == label
    ref_mask = ref == label

    pred_any = pred_mask.any()
    ref_any = ref_mask.any()

    if not pred_any and not ref_any:
        return float("nan")
    if not pred_any or not ref_any:
        return float("inf")

    distances = compute_surface_distances(ref_mask, pred_mask, spacing_mm=spacing)
    return float(compute_robust_hausdorff(distances, 95.0))


def evaluate_case(
    pred_path: Path,
    ref_path: Path,
    case_id: str,
    series_submitter_id: str,
    metrics: set[str] | None = None,
) -> dict:
    """Load one prediction/reference NIfTI pair and compute all requested metrics."""
    if metrics is None:
        metrics = {"dice"}

    ref_img = nib.load(ref_path)
    ref_data = ref_img.get_fdata().astype(np.uint8)
    pred_data = nib.load(pred_path).get_fdata().astype(np.uint8)

    if ref_data.shape != pred_data.shape:
        msg = (
            f"Shape mismatch for {case_id}: "
            f"pred {pred_data.shape} vs ref {ref_data.shape}"
        )
        raise ValueError(msg)

    result: dict = {
        "case_id": case_id,
        "series_submitter_id": series_submitter_id,
    }

    for label_int, label_name in LABELS.items():
        if "dice" in metrics:
            result[f"dice_{label_name}"] = dice_coefficient(pred_data, ref_data, label_int)

        if "hd95" in metrics:
            spacing = ref_img.header.get_zooms()[:3]
            result[f"hd95_{label_name}"] = hausdorff_95(
                pred_data, ref_data, label_int, tuple(float(s) for s in spacing)
            )

    return result


def evaluate_folder(
    pred_dir: Path,
    ref_dir: Path,
    mapping: list[dict],
    output_path: Path | None = None,
    metrics: set[str] | None = None,
    split: str = "test",
) -> pl.DataFrame:
    """Evaluate all matching prediction/reference pairs in the directories.

    Only evaluates cases where both prediction and reference files exist
    and the mapping entry matches the requested split.
    """
    if metrics is None:
        metrics = {"dice"}

    invalid = metrics - VALID_METRICS
    if invalid:
        msg = f"Unknown metrics: {invalid}. Valid: {VALID_METRICS}"
        raise ValueError(msg)

    filtered = [e for e in mapping if e["split"] == split]
    logger.info("Evaluating", split=split, cases=len(filtered), metrics=sorted(metrics))

    results: list[dict] = []
    for i, entry in enumerate(filtered):
        case_id = entry["case_id"]
        pred_path = pred_dir / f"{case_id}.nii.gz"
        ref_path = ref_dir / f"{case_id}.nii.gz"

        if not pred_path.exists():
            logger.warning("Missing prediction", case_id=case_id)
            continue
        if not ref_path.exists():
            logger.warning("Missing reference", case_id=case_id)
            continue

        result = evaluate_case(
            pred_path,
            ref_path,
            case_id,
            entry["series_submitter_id"],
            metrics=metrics,
        )
        results.append(result)

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(filtered)}")

    df = pl.DataFrame(results)
    logger.success("Evaluation complete", cases=df.height)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_csv(output_path)
        logger.info("Saved CSV", path=str(output_path))

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate NIfTI predictions against reference labels"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True, help="case_id_mapping.json")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["dice"],
        choices=sorted(VALID_METRICS),
        help="Metrics to compute (default: dice only)",
    )
    args = parser.parse_args()

    mapping_data = json.loads(args.mapping.read_text())
    evaluate_folder(
        pred_dir=args.predictions,
        ref_dir=args.references,
        mapping=mapping_data,
        output_path=args.output,
        metrics=set(args.metrics),
        split=args.split,
    )
