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

The annotation_file TSV is joined to add NIfTI filenames (needed for nnU-Net
prep and evaluation). It maps series_submitter_id → filename.
"""

import polars as pl

from src.data.exclusions import filter_excluded_cases
from src.data.schemas import ExamSchema
from src.utils.logger import get_logger
from src.utils.settings import settings

logger = get_logger("data.loader")


def _load_annotation_filenames() -> pl.DataFrame:
    """Load annotation_file TSV and return filename → series mapping.

    Filters to image files only (excludes _SEG segmentation masks).
    """
    path = settings.structured_dir / "annotation_file_RSNA_20250321.tsv"
    df = pl.read_csv(path, separator="\t")
    return (
        df.filter(~pl.col("file_name").str.ends_with("_SEG.nii.gz"))
        .select(
            pl.col("file_name").alias("filename"),
            pl.col("mr_series_files.submitter_id").alias("series_submitter_id"),
        )
    )


def load_metadata() -> pl.DataFrame:
    """Load and merge metadata TSVs into one exam-level DataFrame.

    Joins series, study, case, and annotation-file TSVs. Applies
    exclusions and validates the result against ExamSchema.

    Returns:
        Validated Polars DataFrame with one row per exam (~1,254 rows,
        ~71 columns). Key columns are type-checked; extra MIDRC platform
        columns pass through untouched.
    """
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

    annotations = _load_annotation_filenames()

    df = (
        series.join(studies, on="study_submitter_id", how="left")
        .join(cases, on="patient_id", how="left")
        .join(annotations, on="series_submitter_id", how="left")
        .drop(
            "study_submitter_id", "patient_id_right", "cases.submitter_id", "case_ids"
        )
        # Normalize manufacturer casing ("Siemens" → "SIEMENS")
        .with_columns(pl.col("manufacturer").str.to_uppercase())
    )

    df = filter_excluded_cases(df, logger)

    # Validate against schema (extra MIDRC columns pass through)
    df = ExamSchema.validate(df, allow_superfluous_columns=True)

    logger.success("Loaded metadata", rows=df.height, cols=df.width)
    return df
