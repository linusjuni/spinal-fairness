"""
split_v3 — Fully sex-balanced splits via downsampling.

Builds on v2 (race_bin x age_bin x sex stratification), then downsamples
female patients in every split (train, val, test) to match the male count
in that split. Dropped patients are excluded entirely — no data is moved
between splits.

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
    """Return patient IDs to drop from a split to equalise female/male counts."""
    patients = (
        result.filter(pl.col("split") == split)
        .select("patient_id", "sex_bin")
        .unique("patient_id")
    )
    female_ids = patients.filter(pl.col("sex_bin") == "Female")["patient_id"].to_list()
    male_count = patients.filter(pl.col("sex_bin") == "Male").height

    n_drop = len(female_ids) - male_count
    if n_drop > 0:
        drop_ids = set(rng.choice(female_ids, size=n_drop, replace=False).tolist())
        logger.info(
            f"Downsampling females in {split}",
            female_before=len(female_ids),
            male=male_count,
            dropping=n_drop,
            female_after=male_count,
        )
        return drop_ids
    logger.info(f"{split} already balanced — no downsampling needed")
    return set()


def create_splits(df: pl.DataFrame, seed: int | None = None) -> pl.DataFrame:
    """Create fully sex-balanced train/val/test splits.

    Starts from v2 splits, then downsamples female patients in each split
    to match the male count in that split. Dropped patients are excluded.

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

    drop_ids: set = set()
    for split in ("train", "val", "test"):
        drop_ids |= _balance_split(result, split, rng)

    result = result.filter(~pl.col("patient_id").is_in(drop_ids))

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
