from __future__ import annotations

import numpy as np
import polars as pl

from src.data.schemas import Col
from src.utils.logger import get_logger
from src.utils.settings import settings

logger = get_logger(__name__)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20


def log_balance(df: pl.DataFrame) -> None:
    """Log exam counts per stratum x split for verification."""
    balance = (
        df.group_by("stratum", "split")
        .len()
        .pivot(on="split", index="stratum", values="len")
        .fill_null(0)
        .sort("stratum")
    )
    cols = ["stratum"] + [c for c in ("train", "val", "test") if c in balance.columns]
    logger.info("Split balance by stratum:\n" + str(balance.select(cols)))


def save_splits(df: pl.DataFrame, version: str) -> None:
    """Save splits to TSV in settings.splits_dir."""
    path = settings.splits_dir / f"{version}.tsv"
    settings.splits_dir.mkdir(parents=True, exist_ok=True)
    df.write_csv(path, separator="\t")
    logger.success("Saved splits", path=str(path), rows=df.height)


def load_splits(version: str) -> pl.DataFrame:
    """Load splits TSV from settings.splits_dir."""
    path = settings.splits_dir / f"{version}.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"Splits file not found: {path}. Run create_splits() + save_splits() first."
        )
    return pl.read_csv(path, separator="\t")


def apply_splits(df: pl.DataFrame, version: str) -> pl.DataFrame:
    """Join split assignment columns onto an exam-level metadata DataFrame."""
    splits = load_splits(version).select(
        Col.SERIES_SUBMITTER_ID,
        "split",
        "race_bin",
        "age_bin",
        "stratum",
        "annotation_quality",
    )
    return df.join(splits, on=Col.SERIES_SUBMITTER_ID, how="left")


def balance_split_sex(df: pl.DataFrame, split: str, rng: np.random.Generator) -> set:
    """Return series_submitter_ids to drop from a split to equalise male/female counts.

    Drops the majority sex down to match the minority sex count. Works in
    both directions (male-heavy or female-heavy splits).
    """
    split_df = df.filter(pl.col("split") == split)
    male_ids = split_df.filter(pl.col("sex_bin") == "Male")["series_submitter_id"].to_list()
    female_ids = split_df.filter(pl.col("sex_bin") == "Female")["series_submitter_id"].to_list()
    n_male, n_female = len(male_ids), len(female_ids)

    if n_male == n_female:
        logger.info(f"{split} already balanced — no downsampling needed")
        return set()

    if n_male > n_female:
        majority_ids, majority_label, minority_count = male_ids, "Male", n_female
    else:
        majority_ids, majority_label, minority_count = female_ids, "Female", n_male

    n_drop = len(majority_ids) - minority_count
    drop_ids = set(rng.choice(majority_ids, size=n_drop, replace=False).tolist())
    logger.info(
        f"Dropping {n_drop} {majority_label} exams from {split}",
        majority_before=len(majority_ids),
        minority=minority_count,
        dropping=n_drop,
        majority_after=minority_count,
    )
    return drop_ids


# TODO: Run this once we have the list of Gold series_submitter_id values from the Duke authors.
def backfill_annotation_quality(
    gold_series_ids: list[str],
    version: str,
) -> pl.DataFrame:
    """Mark known Gold exams in the splits file and re-save.

    Args:
        gold_series_ids: List of series_submitter_id values that have
                         expert (Gold) annotations.
        version:         Splits version to update.

    Returns:
        Updated splits DataFrame (also re-saved to disk).
    """
    df = load_splits(version)
    df = df.with_columns(
        pl.when(pl.col(Col.SERIES_SUBMITTER_ID).is_in(gold_series_ids))
        .then(pl.lit("gold"))
        .otherwise(pl.lit("silver"))
        .alias("annotation_quality")
    )
    save_splits(df, version)
    logger.success(
        "Backfilled annotation_quality",
        gold=df.filter(pl.col("annotation_quality") == "gold").height,
        silver=df.filter(pl.col("annotation_quality") == "silver").height,
    )
    return df
