"""
Data schema definitions for the CSpineSeg dataset.

Provides:
    - Col: StrEnum of column names referenced in analysis code.
    - Race, Sex, Ethnicity, Manufacturer: StrEnums of valid categorical values.
    - ExamSchema: Patito model that validates the merged exam-level DataFrame.

The ExamSchema is validated once at load time (in loader.py). Downstream code
uses Col and value enums for type-safe, string-free DataFrame access.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Optional

import patito as pt


# ---------------------------------------------------------------------------
# Column name enum (only columns we reference in code)
# ---------------------------------------------------------------------------


class Col(StrEnum):
    """Column names in the merged exam-level DataFrame."""

    # Identifiers
    PATIENT_ID = "patient_id"
    SERIES_SUBMITTER_ID = "series_submitter_id"
    SERIES_UID = "series_uid"
    FILENAME = "filename"

    # Demographics
    RACE = "race"
    SEX = "sex"
    ETHNICITY = "ethnicity"
    AGE = "age_at_imaging"
    AGE_GT89 = "age_at_imaging_gt89"

    # Scanner
    MANUFACTURER = "manufacturer"
    MANUFACTURER_MODEL = "manufacturer_model_name"
    FIELD_STRENGTH = "magnetic_field_strength"
    SLICE_THICKNESS = "slice_thickness"
    PIXEL_SPACING = "pixel_spacing"
    SPACING_BETWEEN_SLICES = "spacing_between_slices"
    ECHO_TIME = "echo_time"
    REPETITION_TIME = "repetition_time"


class VolumeSchema(pt.Model):
    """Schema for the volume properties DataFrame (output of extract_volume_properties)."""

    filename: str = pt.Field(unique=True)
    series_submitter_id: str = pt.Field(unique=True)
    width: int
    height: int
    n_slices: int
    total_voxels: int
    spacing_x: float
    spacing_y: float
    spacing_z: float
    physical_width: float
    physical_height: float
    physical_depth: float
    physical_volume: float
    aspect_ratio_xy: float
    aspect_ratio_xz: float
    aspect_ratio_yz: float
    anisotropy_factor: float


class VolumeCol(StrEnum):
    """Column names in the volume properties DataFrame (output of extract_volume_properties)."""

    # Shape
    WIDTH = "width"
    HEIGHT = "height"
    N_SLICES = "n_slices"
    TOTAL_VOXELS = "total_voxels"

    # Voxel spacing (mm)
    SPACING_X = "spacing_x"
    SPACING_Y = "spacing_y"
    SPACING_Z = "spacing_z"

    # Physical size (mm)
    PHYSICAL_WIDTH = "physical_width"
    PHYSICAL_HEIGHT = "physical_height"
    PHYSICAL_DEPTH = "physical_depth"
    PHYSICAL_VOLUME = "physical_volume"

    # Derived
    ASPECT_RATIO_XY = "aspect_ratio_xy"
    ASPECT_RATIO_XZ = "aspect_ratio_xz"
    ASPECT_RATIO_YZ = "aspect_ratio_yz"
    ANISOTROPY_FACTOR = "anisotropy_factor"


# ---------------------------------------------------------------------------
# Categorical value enums
# ---------------------------------------------------------------------------


class Race(StrEnum):
    """Valid values in the race column."""

    WHITE = "White"
    BLACK = "Black or African American"
    ASIAN = "Asian"
    AMERICAN_INDIAN = "American Indian or Alaska Native"
    PACIFIC_ISLANDER = "Native Hawaiian or other Pacific Islander"
    OTHER = "Other"
    NOT_REPORTED = "Not Reported"


class Sex(StrEnum):
    """Valid values in the sex column."""

    FEMALE = "Female"
    MALE = "Male"


class Ethnicity(StrEnum):
    """Valid values in the ethnicity column."""

    NOT_HISPANIC = "Not Hispanic or Latino"
    HISPANIC = "Hispanic or Latino"
    NOT_REPORTED = "Not Reported"


class Manufacturer(StrEnum):
    """Valid values in the manufacturer column (post-normalization)."""

    SIEMENS = "SIEMENS"
    GE = "GE MEDICAL SYSTEMS"


# ---------------------------------------------------------------------------
# Patito schema — validated at load time
# ---------------------------------------------------------------------------


class ExamSchema(pt.Model):
    """Schema for the merged exam-level DataFrame.

    Only the columns we actively use in analysis are modeled here.
    The full DataFrame has ~70 columns; extra columns pass through
    validation untouched via allow_superfluous_columns=True.

    age_at_imaging is Optional because 13 patients aged >89 have
    their age redacted (null) for privacy, indicated by
    age_at_imaging_gt89 == "Yes".
    """

    # Identifiers
    patient_id: str
    series_submitter_id: str = pt.Field(unique=True)
    series_uid: str
    filename: str = pt.Field(unique=True)

    # Demographics
    race: Literal[
        Race.WHITE,
        Race.BLACK,
        Race.ASIAN,
        Race.AMERICAN_INDIAN,
        Race.PACIFIC_ISLANDER,
        Race.OTHER,
        Race.NOT_REPORTED,
    ]
    sex: Literal[Sex.FEMALE, Sex.MALE]
    ethnicity: Literal[
        Ethnicity.NOT_HISPANIC,
        Ethnicity.HISPANIC,
        Ethnicity.NOT_REPORTED,
    ]
    age_at_imaging: Optional[float]
    age_at_imaging_gt89: str

    # Scanner
    manufacturer: Literal[Manufacturer.SIEMENS, Manufacturer.GE]
    manufacturer_model_name: str
    magnetic_field_strength: float
    slice_thickness: int
    pixel_spacing: float
    spacing_between_slices: float
    echo_time: float
    repetition_time: float
