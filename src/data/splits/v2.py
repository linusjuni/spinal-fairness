"""
split_v2 — Stratified patient-level train/val/test splits with sex stratification.

Stratification axes: race_bin x age_bin x sex  (White and Black patients only)
    race_bin : White | Black
    age_bin  : <40 | 40-60 | 60+   (null age → "60+", all are confirmed >89)
    sex      : Female | Male

Adding sex to the stratification key ensures roughly equal male/female
representation across splits, as required for the sex-based fairness audit.

"Other" race patients are distributed with a simple proportional random split
(no sex stratification), same as v1.

Split unit: patient (not exam). All exams from the same patient land in the
same split to prevent data leakage from the 23 multi-exam patients.

Split ratios: 70% train / 10% val / 20% test.

Output: TSV at settings.splits_dir / "split_v2.tsv"
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.data.groups import age, AgeStrategy, race, RaceStrategy
from src.data.schemas import Col
from src.data.splits.utils import (
    VAL_RATIO,
    TEST_RATIO,
    log_balance,
    save_splits,
    logger,
)
from src.data.loader import load_metadata
from src.utils.settings import settings

VERSION = "split_v2"


def create_splits(df: pl.DataFrame, seed: int | None = None) -> pl.DataFrame:
    """Create stratified patient-level train/val/test splits (race x age x sex).

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

    patients = (
        age[AgeStrategy.THREE_BINS]
        .apply(
            race[RaceStrategy.WHITE_VS_BLACK_VS_OTHER]
            .apply(df, Col.RACE)
            .rename({f"{Col.RACE}_group": "race_bin"}),
            Col.AGE,
        )
        .rename({f"{Col.AGE}_group": "age_bin"})
        .with_columns(
            pl.col(Col.SEX).alias("sex_bin"),
            pl.when(pl.col("race_bin").is_in(["White", "Black"]))
            .then(pl.col("race_bin") + "_" + pl.col("age_bin") + "_" + pl.col(Col.SEX))
            .otherwise(pl.lit("Other"))
            .alias("stratum"),
        )
        .select(Col.PATIENT_ID, "race_bin", "age_bin", "sex_bin", "stratum")
        .unique(subset=[Col.PATIENT_ID], keep="first")
    )

    primary = patients.filter(pl.col("race_bin").is_in(["White", "Black"]))
    other = patients.filter(pl.col("race_bin") == "Other")

    assignments: list[dict] = []

    for stratum in sorted(primary["stratum"].unique().to_list()):
        group = primary.filter(pl.col("stratum") == stratum)
        patient_ids = group[Col.PATIENT_ID].to_list()
        n = len(patient_ids)
        shuffled = rng.permutation(patient_ids).tolist()
        n_test = max(1, round(n * TEST_RATIO))
        n_val = max(1, round(n * VAL_RATIO))
        n_train = n - n_test - n_val
        for pid, split in zip(
            shuffled, ["test"] * n_test + ["val"] * n_val + ["train"] * n_train
        ):
            assignments.append({"patient_id": pid, "split": split})
        logger.info(f"Stratum {stratum!r}", n=n, train=n_train, val=n_val, test=n_test)

    other_ids = rng.permutation(other[Col.PATIENT_ID].to_list()).tolist()
    n = len(other_ids)
    n_test = max(1, round(n * TEST_RATIO))
    n_val = max(1, round(n * VAL_RATIO))
    n_train = n - n_test - n_val
    for pid, split in zip(
        other_ids, ["test"] * n_test + ["val"] * n_val + ["train"] * n_train
    ):
        assignments.append({"patient_id": pid, "split": split})
    logger.info(
        "Stratum 'Other' (unstratified)", n=n, train=n_train, val=n_val, test=n_test
    )

    split_df = pl.DataFrame(assignments)

    result = (
        age[AgeStrategy.THREE_BINS]
        .apply(
            race[RaceStrategy.WHITE_VS_BLACK_VS_OTHER]
            .apply(df, Col.RACE)
            .rename({f"{Col.RACE}_group": "race_bin"}),
            Col.AGE,
        )
        .rename({f"{Col.AGE}_group": "age_bin"})
    )
    result = (
        result.with_columns(
            pl.col(Col.SEX).alias("sex_bin"),
            pl.when(pl.col("race_bin").is_in(["White", "Black"]))
            .then(pl.col("race_bin") + "_" + pl.col("age_bin") + "_" + pl.col(Col.SEX))
            .otherwise(pl.lit("Other"))
            .alias("stratum"),
        )
        .join(split_df, on=Col.PATIENT_ID, how="left")
        .select(
            Col.PATIENT_ID,
            Col.SERIES_SUBMITTER_ID,
            "split",
            "race_bin",
            "age_bin",
            "sex_bin",
            "stratum",
            pl.lit(None).cast(pl.String).alias("annotation_quality"),
        )
    )

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
