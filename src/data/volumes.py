from __future__ import annotations

import polars as pl

from src.utils.logger import get_logger
from src.utils.settings import settings

logger = get_logger("data.volumes")


def load_annotation_file() -> pl.DataFrame:
    """
    Load annotation_file TSV and filter to image files only.

    The annotation file contains 2,510 rows (1,255 images + 1,255 segmentations).
    This function filters to only the image files by excluding files ending with
    "_SEG.nii.gz".

    Returns:
        DataFrame with columns:
            - filename (str): NIfTI filename without path
            - series_submitter_id (str): Series UID for joining with metadata

        Shape: 1,255 rows (images only)

    Raises:
        FileNotFoundError: If annotation file TSV doesn't exist
    """
    path = settings.structured_dir / "annotation_file_RSNA_20250321.tsv"

    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    df = pl.read_csv(path, separator="\t")

    # Filter to images only (exclude segmentations ending with _SEG.nii.gz)
    df = df.filter(~pl.col("file_name").str.ends_with("_SEG.nii.gz"))

    logger.success("Loaded annotation file", rows=df.height)

    # Return only the columns we need, with renamed columns
    return df.select([
        pl.col("file_name").alias("filename"),
        pl.col("mr_series_files.submitter_id").alias("series_submitter_id"),
    ])


def extract_volume_properties(force_refresh: bool = False) -> pl.DataFrame:
    """
    Extract volume properties from all NIfTI files, with Parquet caching.

    Reads NIfTI headers (NOT full volumes) for all 1,255 image files and
    extracts shape (width, height, n_slices) and spacing (voxel mm) information.
    Results are cached to Parquet for fast subsequent loads.

    Cache is invalidated if:
        - force_refresh=True
        - Cache file doesn't exist
        - Cache file is older than annotation_file TSV (staleness check)

    Args:
        force_refresh: If True, bypass cache and recompute properties

    Returns:
        DataFrame with columns:
            - filename (str)
            - series_submitter_id (str)
            - width, height, n_slices (int) - voxel dimensions
            - total_voxels (int)
            - spacing_x, spacing_y, spacing_z (float) - mm per voxel
            - physical_width, physical_height, physical_depth (float) - mm
            - physical_volume (float) - mm³
            - aspect_ratio_xy, aspect_ratio_xz, aspect_ratio_yz (float)
            - anisotropy_factor (float)

        Shape: 1,255 rows

    Raises:
        FileNotFoundError: If annotation directory doesn't exist
        ValueError: If critical NIfTI loading errors prevent extraction
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
    annotation_df = load_annotation_file()
    logger.info("Starting volume property extraction", files=annotation_df.height)

    # Import nibabel here (only needed for extraction, not for cache loading)
    import nibabel as nib

    properties = []
    failed_files = []

    for i, row in enumerate(annotation_df.iter_rows(named=True)):
        filename = row["filename"]
        series_submitter_id = row["series_submitter_id"]

        # Progress logging
        if i % 100 == 0:
            logger.info(f"Processing {i}/{annotation_df.height}")

        path = settings.annotation_dir / filename

        # Check file exists
        if not path.exists():
            logger.warning(f"File not found", filename=filename)
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
                    f"Unexpected shape",
                    filename=filename,
                    shape=shape,
                    action="skipping",
                )
                failed_files.append((filename, f"Unexpected shape: {shape}"))
                continue

            # Validate spacing information exists
            if len(spacing) < 3:
                logger.warning(
                    f"Missing spacing information", filename=filename, action="skipping"
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
                    "filename": filename,
                    "series_submitter_id": series_submitter_id,
                    "width": int(width),
                    "height": int(height),
                    "n_slices": int(n_slices),
                    "total_voxels": int(total_voxels),
                    "spacing_x": float(spacing_x),
                    "spacing_y": float(spacing_y),
                    "spacing_z": float(spacing_z),
                    "physical_width": float(physical_width),
                    "physical_height": float(physical_height),
                    "physical_depth": float(physical_depth),
                    "physical_volume": float(physical_volume),
                    "aspect_ratio_xy": float(aspect_ratio_xy),
                    "aspect_ratio_xz": float(aspect_ratio_xz),
                    "aspect_ratio_yz": float(aspect_ratio_yz),
                    "anisotropy_factor": float(anisotropy_factor),
                }
            )

        except Exception as e:
            logger.warning(f"Failed to load", filename=filename, error=str(e))
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

    # Cache to Parquet
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_path)
    logger.success(
        "Cached volume properties", rows=df.height, path=str(cache_path.name)
    )

    return df


def load_volume_properties(force_refresh: bool = False) -> pl.DataFrame:
    """
    Load volume properties, extracting if cache is stale or missing.

    This is the main public API for accessing volume properties. It handles
    caching logic transparently - first checking for a valid cache, and only
    extracting from NIfTI files if necessary.

    Args:
        force_refresh: If True, bypass cache and recompute properties

    Returns:
        DataFrame with volume properties
        Shape: 1,255 rows × 18 columns

    Example:
        >>> from src.data.volumes import load_volume_properties
        >>> volumes = load_volume_properties()
        >>> print(volumes.shape)
        (1255, 18)
        >>> print(volumes.columns[:5])
        ['filename', 'series_submitter_id', 'width', 'height', 'n_slices']
    """
    return extract_volume_properties(force_refresh)
