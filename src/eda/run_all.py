"""
Run all EDA scripts across cohorts: full dataset, train, val, and test splits.

Loads each data source once, then filters per cohort and delegates to each
EDA module's run() function. Outputs land in:

    outputs/eda/{cohort}/{analysis}/TIMESTAMP/

where cohort is one of: full, train, val, test.

Usage:
    uv run -m src.eda.run_all
"""

import polars as pl

from src.data.loader import load_metadata
from src.data.mri_volumes import load_mri_volume_properties
from src.data.schemas import Col
from src.data.segmentation_volumes import load_segmentation_volumes
from src.data.splits.utils import load_splits
from src.eda import crosscuts, demographics, mri_slices, mri_volumes, scanner, segmentation_volumes
from src.utils.logger import get_logger

logger = get_logger(__name__)

COHORTS = ["full", "train", "val", "test"]
SPLIT_VERSION = "split_v3"


def main() -> None:
    logger.info("Loading data sources")
    metadata = load_metadata()
    mri_vols = load_mri_volume_properties()
    seg_vols = load_segmentation_volumes()
    splits = load_splits(SPLIT_VERSION)

    mri_df = mri_vols.join(metadata, on=Col.SERIES_SUBMITTER_ID, how="left")
    seg_df = seg_vols.join(metadata, on=Col.SERIES_SUBMITTER_ID, how="left")

    for cohort in COHORTS:
        logger.info(f"Running EDA for cohort: {cohort}")

        if cohort == "full":
            meta_c = metadata
            mri_c = mri_df
            seg_c = seg_df
        else:
            ids = splits.filter(pl.col("split") == cohort)[Col.SERIES_SUBMITTER_ID].to_list()
            meta_c = metadata.filter(pl.col(Col.SERIES_SUBMITTER_ID).is_in(ids))
            mri_c = mri_df.filter(pl.col(Col.SERIES_SUBMITTER_ID).is_in(ids))
            seg_c = seg_df.filter(pl.col(Col.SERIES_SUBMITTER_ID).is_in(ids))

        logger.info(f"Cohort size: {meta_c.height} exams")

        demographics.run(meta_c, f"{cohort}/demographics")
        scanner.run(meta_c, f"{cohort}/scanner")
        crosscuts.run(meta_c, f"{cohort}/crosscuts")
        mri_volumes.run(mri_c, f"{cohort}/mri_volumes")
        mri_slices.run(mri_c, f"{cohort}/mri_slices")
        segmentation_volumes.run(seg_c, f"{cohort}/segmentation_volumes")

    logger.info("All EDA runs complete")


if __name__ == "__main__":
    main()
