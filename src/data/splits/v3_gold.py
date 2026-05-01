"""
split_v3_gold — Sex-balanced gold-only subset of split_v3.

Takes the gold-labelled cases from split_v3 (annotation_quality == "gold")
and re-applies sex-balancing within each split so every split is exactly
50/50 male/female within the gold cohort.

This is the training split for the gold-label experiment (Dataset002).
The test set (76 cases) uses only expert-annotated labels, making it the
ground-truth reference for the gold vs silver label comparison.

Output: TSV at settings.splits_dir / "split_v3_gold.tsv"
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

VERSION = "split_v3_gold"


def create_splits(seed: int | None = None) -> pl.DataFrame:
    """Filter split_v3 to gold cases and re-balance sex within each split.

    Args:
        seed: Random seed. Defaults to settings.RANDOM_SEED.

    Returns:
        DataFrame with same columns as split_v3, filtered to gold cases
        with 50/50 sex balance enforced within each split.
    """
    if seed is None:
        seed = settings.RANDOM_SEED

    rng = np.random.default_rng(seed)

    df = load_splits("split_v3").filter(pl.col("annotation_quality") == "gold")
    logger.info("Gold cases loaded from split_v3", rows=df.height)

    drop_ids: set = set()
    for split in ("train", "val", "test"):
        drop_ids |= balance_split_sex(df, split, rng)

    result = df.filter(~pl.col("series_submitter_id").is_in(drop_ids))
    log_balance(result)
    return result


if __name__ == "__main__":
    splits = create_splits()
    save_splits(splits, VERSION)
