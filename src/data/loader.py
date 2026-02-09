"""
Join logic:
All three TSVs have a "submitter_id" column, but it means something different
in each:
    - patient ID in case_RSNA_20250321.tsv
    - study UID in imaging_study_RSNA_20250321.tsv
    - series UID in mr_series_RSNA_20250321.tsv

The foreign keys that link them are:
    - series["imaging_studies.submitter_id"]  ==  study["submitter_id"]   (study UIDs)
    - study["case_ids"]                       ==  case["submitter_id"]    (patient IDs)
We rename these to a shared name (study_submitter_id / patient_id) so polars
can join on a single column. Columns that collide (type, case_ids,
image_data_modified) are prefixed per table. Redundant copies of join keys
are dropped after merging.
"""

import polars as pl

from src.utils.settings import settings
from src.utils.logger import get_logger

logger = get_logger("data.loader")


def load_metadata() -> pl.DataFrame:
    """Load and merge the three metadata TSVs into one exam-level DataFrame."""
    d = settings.structured_dir

    cases = pl.read_csv(d / "case_RSNA_20250321.tsv", separator="\t").rename(
        {"submitter_id": "patient_id", "type": "case_type"}
    )

    studies = pl.read_csv(d / "imaging_study_RSNA_20250321.tsv", separator="\t").rename(
        {
            "submitter_id": "study_submitter_id",
            "case_ids": "patient_id",
            "type": "study_type",
        }
    )

    series = pl.read_csv(d / "mr_series_RSNA_20250321.tsv", separator="\t").rename(
        {
            "imaging_studies.submitter_id": "study_submitter_id",
            "type": "series_type",
            "submitter_id": "series_submitter_id",
            "case_ids": "patient_id",
            "image_data_modified": "series_image_data_modified",
        }
    )

    df = (
        series.join(studies, on="study_submitter_id", how="left")
        .join(cases, on="patient_id", how="left")
        .drop(
            "study_submitter_id", "patient_id_right", "cases.submitter_id", "case_ids"
        )
        # Normalize manufacturer casing ("Siemens" → "SIEMENS")
        .with_columns(pl.col("manufacturer").str.to_uppercase())
    )

    logger.success("Loaded metadata", rows=df.height, cols=df.width)
    return df
