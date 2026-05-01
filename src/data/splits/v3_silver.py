"""
split_v3_silver — Sex-balanced silver-only subset of split_v3.

Takes the silver-labelled cases from split_v3 (annotation_quality == "silver")
and re-applies sex-balancing within each split so every split is exactly
50/50 male/female within the silver cohort.

This is the training split for the silver-label experiment (Dataset003).
Evaluating this model against the gold test set (from split_v3_gold) reveals
the true fairness gap; evaluating against the silver test set reveals the
Biased Ruler effect.

Output: TSV at settings.splits_dir / "split_v3_silver.tsv"
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.data.splits.utils import (
    balance_split_sex,
    load_splits,
    log_balance,
    logger,
    save_splits,
)
from src.utils.settings import settings

VERSION = "split_v3_silver"


def create_splits(seed: int | None = None) -> pl.DataFrame:
    """Filter split_v3 to silver cases and re-balance sex within each split.

    Args:
        seed: Random seed. Defaults to settings.RANDOM_SEED.

    Returns:
        DataFrame with same columns as split_v3, filtered to silver cases
        with 50/50 sex balance enforced within each split.
    """
    if seed is None:
        seed = settings.RANDOM_SEED

    rng = np.random.default_rng(seed)

    df = load_splits("split_v3").filter(pl.col("annotation_quality") == "silver")
    logger.info("Silver cases loaded from split_v3", rows=df.height)

    drop_ids: set = set()
    for split in ("train", "val", "test"):
        drop_ids |= balance_split_sex(df, split, rng)

    result = df.filter(~pl.col("series_submitter_id").is_in(drop_ids))
    log_balance(result)
    return result


if __name__ == "__main__":
    splits = create_splits()
    save_splits(splits, VERSION)
