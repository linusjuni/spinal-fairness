from __future__ import annotations

import polars as pl

from src.data.schemas import Col
from src.utils.logger import get_logger
from src.utils.settings import settings

logger = get_logger("data.splits")

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
