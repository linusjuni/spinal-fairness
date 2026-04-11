from __future__ import annotations

import polars as pl
import nibabel as nib

from src.data.exclusions import filter_excluded_cases
from src.data.loader import load_annotation_filenames
from src.data.schemas import Col, VolumeCol, VolumeSchema
from src.utils.logger import get_logger
from src.utils.settings import settings

logger = get_logger(__name__)


def extract_mri_volume_properties(force_refresh: bool = False) -> pl.DataFrame:
    """
    Extract volume properties from all NIfTI files, with Parquet caching.

    Reads NIfTI headers (NOT full volumes) for all 1,255 image files and
    extracts shape (width, height, n_slices) and spacing (voxel mm) information.
    Results are cached to Parquet for fast subsequent loads.

    Cache is invalidated if:
        - force_refresh=True
        - Cache file doesn't exist
        - Cache file is older than annotation_file TSV (staleness check)
    """
    cache_path = settings.processed_dir / "volume_properties.parquet"

    # Cache staleness check
    if not force_refresh and cache_path.exists():
        annotation_tsv = settings.structured_dir / "annotation_file_RSNA_20250321.tsv"

        if cache_path.stat().st_mtime > annotation_tsv.stat().st_mtime:
            try:
                df = pl.read_parquet(cache_path)
                logger.info("Loaded from cache", rows=df.height)
                return df
            except Exception as e:
                logger.warning("Cache corrupted, recomputing", error=str(e))
                cache_path.unlink(missing_ok=True)
                # Fall through to extraction

    # Check annotation directory exists
    if not settings.annotation_dir.exists():
        raise FileNotFoundError(
            f"Annotation directory not found: {settings.annotation_dir}"
        )

    # Load annotation file to get filename -> series_submitter_id mapping
    annotation_df = load_annotation_filenames()
    logger.info("Starting volume property extraction", files=annotation_df.height)

    properties = []
    failed_files = []

    for i, row in enumerate(annotation_df.iter_rows(named=True)):
        filename = row[Col.FILENAME]
        series_submitter_id = row[Col.SERIES_SUBMITTER_ID]

        # Progress logging
        if i % 100 == 0:
            logger.info(f"Processing {i}/{annotation_df.height}")

        path = settings.annotation_dir / filename

        # Check file exists
        if not path.exists():
            logger.warning("File not found", filename=filename)
            failed_files.append((filename, "File not found on disk"))
            continue

        try:
            # Load NIfTI header (lazy loading - doesn't read voxel data)
            img = nib.load(path)

            shape = img.shape  # (width, height, n_slices)
            spacing = img.header.get_zooms()  # (spacing_x, spacing_y, spacing_z)

            # Validate shape is 3D
            if len(shape) != 3:
                logger.warning(
                    "Unexpected shape",
                    filename=filename,
                    shape=shape,
                    action="skipping",
                )
                failed_files.append((filename, f"Unexpected shape: {shape}"))
                continue

            # Validate spacing information exists
            if len(spacing) < 3:
                logger.warning(
                    "Missing spacing information", filename=filename, action="skipping"
                )
                failed_files.append((filename, "Missing spacing information"))
                continue

            # Extract properties
            width, height, n_slices = shape[0], shape[1], shape[2]
            spacing_x, spacing_y, spacing_z = spacing[0], spacing[1], spacing[2]

            # Compute derived properties
            total_voxels = width * height * n_slices
            physical_width = width * spacing_x
            physical_height = height * spacing_y
            physical_depth = n_slices * spacing_z
            physical_volume = physical_width * physical_height * physical_depth

            aspect_ratio_xy = width / height
            aspect_ratio_xz = width / n_slices
            aspect_ratio_yz = height / n_slices

            # Anisotropy: ratio of slice thickness to in-plane resolution
            anisotropy_factor = spacing_z / (spacing_x * spacing_y) ** 0.5

            properties.append(
                {
                    Col.FILENAME: filename,
                    Col.SERIES_SUBMITTER_ID: series_submitter_id,
                    VolumeCol.WIDTH: int(width),
                    VolumeCol.HEIGHT: int(height),
                    VolumeCol.N_SLICES: int(n_slices),
                    VolumeCol.TOTAL_VOXELS: int(total_voxels),
                    VolumeCol.SPACING_X: float(spacing_x),
                    VolumeCol.SPACING_Y: float(spacing_y),
                    VolumeCol.SPACING_Z: float(spacing_z),
                    VolumeCol.PHYSICAL_WIDTH: float(physical_width),
                    VolumeCol.PHYSICAL_HEIGHT: float(physical_height),
                    VolumeCol.PHYSICAL_DEPTH: float(physical_depth),
                    VolumeCol.PHYSICAL_VOLUME: float(physical_volume),
                    VolumeCol.ASPECT_RATIO_XY: float(aspect_ratio_xy),
                    VolumeCol.ASPECT_RATIO_XZ: float(aspect_ratio_xz),
                    VolumeCol.ASPECT_RATIO_YZ: float(aspect_ratio_yz),
                    VolumeCol.ANISOTROPY_FACTOR: float(anisotropy_factor),
                }
            )

        except Exception as e:
            logger.warning("Failed to load", filename=filename, error=str(e))
            failed_files.append((filename, str(e)))

    # Log failed files summary
    if failed_files:
        logger.warning(
            f"Failed to load {len(failed_files)} files",
            success_rate=f"{100 * len(properties) / annotation_df.height:.1f}%",
        )

    # Create DataFrame
    df = pl.DataFrame(properties)

    if df.height == 0:
        raise ValueError("No properties extracted - all files failed to load")

    # Validate schema before caching
    VolumeSchema.validate(df)

    # Cache to Parquet
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_path)
    logger.success(
        "Cached volume properties", rows=df.height, path=str(cache_path.name)
    )

    return df


def load_mri_volume_properties(force_refresh: bool = False) -> pl.DataFrame:
    """Load volume properties, extracting if cache is stale or missing."""
    df = extract_mri_volume_properties(force_refresh)
    df = filter_excluded_cases(df, logger)
    return df
