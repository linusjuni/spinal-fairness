"""
Stratified patient-level train/val/test splits for the CSpineSeg fairness audit.

Stratification axes: race_bin x age_bin  (White and Black patients only)
    race_bin : White | Black
    age_bin  : <40 | 40-60 | 60+   (null age → "60+", all are confirmed >89)

"Other" race patients (Asian, American Indian or Alaska Native, Pacific Islander, Not Reported) are
distributed with a simple proportional random split. They are included as
training data but excluded from the White vs Black fairness comparison.
Their stratum is recorded as "Other" (no age suffix) to signal this.

Split unit: patient (not exam). All exams from the same patient land in the
same split to prevent data leakage from the 23 multi-exam patients.

Split ratios: 70% train / 10% val / 20% test.

The `annotation_quality` column is null until Gold/Silver case IDs are
received from the Duke authors. Once known, run backfill_annotation_quality()
and re-save. At evaluation time filter on annotation_quality to choose
which subset of the test set to score against.

Output: TSV at settings.splits_dir / "split_v1.tsv"
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.data.groups import age, AgeStrategy, race, RaceStrategy
from src.data.schemas import Col

from src.utils.logger import get_logger
from src.utils.settings import settings

logger = get_logger("data.splits")

SPLIT_VERSION = "split_v1"
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20


def create_splits(df: pl.DataFrame, seed: int | None = None) -> pl.DataFrame:
    """Create stratified patient-level train/val/test splits.

    Args:
        df:   Exam-level metadata DataFrame from load_metadata().
        seed: Random seed. Defaults to settings.RANDOM_SEED.

    Returns:
        Exam-level DataFrame with columns:
            patient_id, series_submitter_id, split,
            race_bin, age_bin, stratum, annotation_quality
    """
    if seed is None:
        seed = settings.RANDOM_SEED

    rng = np.random.default_rng(seed)

    # Add bins at exam level, then collapse to one row per patient.
    # For multi-exam patients all exams share the same demographics, so
    # keep="first" is safe.
    patients = age[AgeStrategy.THREE_BINS].apply(
        race[RaceStrategy.WHITE_VS_BLACK_VS_OTHER].apply(df, Col.RACE).rename({f"{Col.RACE}_group": "race_bin"}),
        Col.AGE,
    ).rename({f"{Col.AGE}_group": "age_bin"}).with_columns(
        pl.when(pl.col("race_bin").is_in(["White", "Black"]))
        .then(pl.col("race_bin") + "_" + pl.col("age_bin"))
        .otherwise(pl.lit("Other"))
        .alias("stratum")
    ).select(Col.PATIENT_ID, "race_bin", "age_bin", "stratum").unique(
        subset=[Col.PATIENT_ID], keep="first"
    )

    # White + Black: stratified on race_bin x age_bin
    primary = patients.filter(pl.col("race_bin").is_in(["White", "Black"]))
    other = patients.filter(pl.col("race_bin") == "Other")

    assignments: list[dict] = []

    # Iterate strata in sorted order so the same seed always produces the
    # same split regardless of polars' internal group_by ordering.
    strata = sorted(primary["stratum"].unique().to_list())

    for stratum in strata:
        group = primary.filter(pl.col("stratum") == stratum)
        patient_ids = group[Col.PATIENT_ID].to_list()
        n = len(patient_ids)

        shuffled = rng.permutation(patient_ids).tolist()

        n_test = max(1, round(n * TEST_RATIO))
        n_val = max(1, round(n * VAL_RATIO))
        n_train = n - n_test - n_val

        for pid, split in zip(shuffled, ["test"] * n_test + ["val"] * n_val + ["train"] * n_train):
            assignments.append({"patient_id": pid, "split": split})

        logger.info(
            f"Stratum {stratum!r}",
            n=n, train=n_train, val=n_val, test=n_test,
        )

    # Other: simple proportional random split, no stratification
    other_ids = rng.permutation(other[Col.PATIENT_ID].to_list()).tolist()
    n = len(other_ids)
    n_test = max(1, round(n * TEST_RATIO))
    n_val = max(1, round(n * VAL_RATIO))
    n_train = n - n_test - n_val

    for pid, split in zip(other_ids, ["test"] * n_test + ["val"] * n_val + ["train"] * n_train):
        assignments.append({"patient_id": pid, "split": split})

    logger.info("Stratum 'Other' (unstratified)", n=n, train=n_train, val=n_val, test=n_test)

    split_df = pl.DataFrame(assignments)

    # Join split assignment back to exam level, then attach bin columns.
    result = age[AgeStrategy.THREE_BINS].apply(
        race[RaceStrategy.WHITE_VS_BLACK_VS_OTHER].apply(df, Col.RACE).rename({f"{Col.RACE}_group": "race_bin"}),
        Col.AGE,
    ).rename({f"{Col.AGE}_group": "age_bin"})
    result = result.with_columns(
        pl.when(pl.col("race_bin").is_in(["White", "Black"]))
        .then(pl.col("race_bin") + "_" + pl.col("age_bin"))
        .otherwise(pl.lit("Other"))
        .alias("stratum")
    ).join(split_df, on=Col.PATIENT_ID, how="left").select(
        Col.PATIENT_ID,
        Col.SERIES_SUBMITTER_ID,
        "split",
        "race_bin",
        "age_bin",
        "stratum",
        pl.lit(None).cast(pl.String).alias("annotation_quality"),
    )

    _log_balance(result)
    return result


def _log_balance(df: pl.DataFrame) -> None:
    """Log exam counts per stratum x split for verification."""
    balance = (
        df.group_by("stratum", "split")
        .len()
        .pivot(on="split", index="stratum", values="len")
        .fill_null(0)
        .sort("stratum")
    )
    # Ensure consistent column order in log output
    cols = ["stratum"] + [c for c in ("train", "val", "test") if c in balance.columns]
    logger.info("Split balance by stratum:\n" + str(balance.select(cols)))


def summarise_splits(df: pl.DataFrame) -> pl.DataFrame:
    """Return a readable summary of split balance by race_bin and age_bin.

    Useful for verifying that the test set has sufficient representation
    for the primary fairness comparison (White vs Black).
    """
    return (
        df.group_by("race_bin", "age_bin", "split")
        .len()
        .pivot(on="split", index=["race_bin", "age_bin"], values="len")
        .fill_null(0)
        .sort("race_bin", "age_bin")
    )


def save_splits(df: pl.DataFrame, version: str = SPLIT_VERSION) -> None:
    """Save splits to TSV in settings.splits_dir."""
    path = settings.splits_dir / f"{version}.tsv"
    settings.splits_dir.mkdir(parents=True, exist_ok=True)
    df.write_csv(path, separator="\t")
    logger.success("Saved splits", path=str(path), rows=df.height)


def load_splits(version: str = SPLIT_VERSION) -> pl.DataFrame:
    """Load splits TSV from settings.splits_dir."""
    path = settings.splits_dir / f"{version}.tsv"
    if not path.exists():
        raise FileNotFoundError(
            f"Splits file not found: {path}. Run create_splits() + save_splits() first."
        )
    return pl.read_csv(path, separator="\t")


def apply_splits(df: pl.DataFrame, version: str = SPLIT_VERSION) -> pl.DataFrame:
    """Join split assignment columns onto an exam-level metadata DataFrame."""
    splits = load_splits(version).select(
        Col.SERIES_SUBMITTER_ID, "split", "race_bin", "age_bin", "stratum", "annotation_quality"
    )
    return df.join(splits, on=Col.SERIES_SUBMITTER_ID, how="left")


# TODO: Run this once we have the list of Gold series_submitter_id values from the Duke authors.
def backfill_annotation_quality(
    gold_series_ids: list[str],
    version: str = SPLIT_VERSION,
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


if __name__ == "__main__":
    from src.data.loader import load_metadata

    df = load_metadata()
    splits = create_splits(df)
    save_splits(splits)

    summary = summarise_splits(splits)
    logger.info("\nSplit balance (race_bin  age_bin):")
    logger.info(summary)
