"""
split_v3 — Fully sex-balanced splits via exam-level downsampling.

Builds on v2 (race_bin x age_bin x sex stratification), then drops female
exams in every split (train, val, test) until the female exam count exactly
matches the male exam count. Balancing operates at the exam level: for
multi-exam patients, some exams may be dropped while others are retained.
Since v2 already confines all exams from a patient to the same split, this
does not introduce data leakage.

This mirrors Aditya et al.'s approach of using a fully balanced cohort for
both training and evaluation, enabling controlled experiments to test whether
representational imbalance causes observed fairness gaps.

Output: TSV at settings.splits_dir / "split_v3.tsv"
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.data.loader import load_metadata
from src.data.splits.utils import (
    log_balance,
    save_splits,
    logger,
)
from src.data.splits.v2 import create_splits as _create_splits_v2
from src.utils.settings import settings

VERSION = "split_v3"


def _balance_split(result: pl.DataFrame, split: str, rng: np.random.Generator) -> set:
    """Return series_submitter_ids to drop from a split to equalise female/male exam counts."""
    split_df = result.filter(pl.col("split") == split)
    male_exam_count = split_df.filter(pl.col("sex_bin") == "Male").height
    female_exam_ids = split_df.filter(pl.col("sex_bin") == "Female")["series_submitter_id"].to_list()

    n_drop = len(female_exam_ids) - male_exam_count
    if n_drop > 0:
        drop_ids = set(rng.choice(female_exam_ids, size=n_drop, replace=False).tolist())
        logger.info(
            f"Dropping {n_drop} female exams from {split}",
            female_exams_before=len(female_exam_ids),
            male_exams=male_exam_count,
            dropping=n_drop,
            female_exams_after=male_exam_count,
        )
        return drop_ids
    logger.info(f"{split} already balanced — no downsampling needed")
    return set()


def create_splits(df: pl.DataFrame, seed: int | None = None) -> pl.DataFrame:
    """Create fully sex-balanced train/val/test splits.

    Starts from v2 splits, then drops female exams in each split until the
    female exam count exactly matches the male exam count. For multi-exam
    patients, some exams may be retained while others are dropped.

    Args:
        df:   Exam-level metadata DataFrame from load_metadata().
        seed: Random seed. Defaults to settings.RANDOM_SEED.

    Returns:
        Exam-level DataFrame with columns:
            patient_id, series_submitter_id, split,
            race_bin, age_bin, sex_bin, stratum, annotation_quality
    """
    if seed is None:
        seed = settings.RANDOM_SEED

    rng = np.random.default_rng(seed)

    result = _create_splits_v2(df, seed=seed)

    drop_exam_ids: set = set()
    for split in ("train", "val", "test"):
        drop_exam_ids |= _balance_split(result, split, rng)

    result = result.filter(~pl.col("series_submitter_id").is_in(drop_exam_ids))

    log_balance(result)
    return result


def summarise_splits(df: pl.DataFrame) -> pl.DataFrame:
    """Return a readable summary of split balance by race_bin, age_bin, and sex_bin."""
    return (
        df.group_by("race_bin", "age_bin", "sex_bin", "split")
        .len()
        .pivot(on="split", index=["race_bin", "age_bin", "sex_bin"], values="len")
        .fill_null(0)
        .sort("race_bin", "age_bin", "sex_bin")
    )


if __name__ == "__main__":
    df = load_metadata()
    splits = create_splits(df)
    save_splits(splits, VERSION)

    logger.info("\nSplit balance (race_bin x age_bin x sex_bin):")
    logger.info(summarise_splits(splits))
