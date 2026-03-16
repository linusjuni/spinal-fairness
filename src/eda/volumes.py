import polars as pl
import seaborn as sns

from src.data.loader import load_metadata
from src.data.schemas import Col, VolumeCol
from src.data.volumes import load_volume_properties
from src.eda.report import EDAReport

sns.set_theme(style="whitegrid", palette="muted")

# Load data
volumes = load_volume_properties(force_refresh=False)
metadata = load_metadata()

# Merge on series_submitter_id
df = volumes.join(metadata, on=Col.SERIES_SUBMITTER_ID, how="left")

with EDAReport("volumes") as report:
    # 1. Summary statistics
    report.log_stat("total_images", df.height)
    report.log_stat(
        "shape_stats",
        {
            VolumeCol.WIDTH: {
                "mean": df[VolumeCol.WIDTH].mean(),
                "std": df[VolumeCol.WIDTH].std(),
                "min": df[VolumeCol.WIDTH].min(),
                "max": df[VolumeCol.WIDTH].max(),
            },
            VolumeCol.HEIGHT: {
                "mean": df[VolumeCol.HEIGHT].mean(),
                "std": df[VolumeCol.HEIGHT].std(),
                "min": df[VolumeCol.HEIGHT].min(),
                "max": df[VolumeCol.HEIGHT].max(),
            },
            VolumeCol.N_SLICES: {
                "mean": df[VolumeCol.N_SLICES].mean(),
                "std": df[VolumeCol.N_SLICES].std(),
                "min": df[VolumeCol.N_SLICES].min(),
                "max": df[VolumeCol.N_SLICES].max(),
            },
        },
    )
    report.log_stat(
        "spacing_stats",
        {
            VolumeCol.SPACING_X: {
                "mean": df[VolumeCol.SPACING_X].mean(),
                "std": df[VolumeCol.SPACING_X].std(),
            },
            VolumeCol.SPACING_Y: {
                "mean": df[VolumeCol.SPACING_Y].mean(),
                "std": df[VolumeCol.SPACING_Y].std(),
            },
            VolumeCol.SPACING_Z: {
                "mean": df[VolumeCol.SPACING_Z].mean(),
                "std": df[VolumeCol.SPACING_Z].std(),
            },
        },
    )

    # Save summary table
    summary = df.select(
        [
            pl.col(VolumeCol.WIDTH).mean().alias("mean_width"),
            pl.col(VolumeCol.WIDTH).std().alias("std_width"),
            pl.col(VolumeCol.WIDTH).min().alias("min_width"),
            pl.col(VolumeCol.WIDTH).max().alias("max_width"),
            pl.col(VolumeCol.HEIGHT).mean().alias("mean_height"),
            pl.col(VolumeCol.HEIGHT).std().alias("std_height"),
            pl.col(VolumeCol.HEIGHT).min().alias("min_height"),
            pl.col(VolumeCol.HEIGHT).max().alias("max_height"),
            pl.col(VolumeCol.N_SLICES).mean().alias("mean_n_slices"),
            pl.col(VolumeCol.N_SLICES).std().alias("std_n_slices"),
            pl.col(VolumeCol.N_SLICES).min().alias("min_n_slices"),
            pl.col(VolumeCol.N_SLICES).max().alias("max_n_slices"),
            pl.col(VolumeCol.SPACING_X).mean().alias("mean_spacing_x"),
            pl.col(VolumeCol.SPACING_X).std().alias("std_spacing_x"),
            pl.col(VolumeCol.SPACING_Y).mean().alias("mean_spacing_y"),
            pl.col(VolumeCol.SPACING_Y).std().alias("std_spacing_y"),
            pl.col(VolumeCol.SPACING_Z).mean().alias("mean_spacing_z"),
            pl.col(VolumeCol.SPACING_Z).std().alias("std_spacing_z"),
        ]
    )
    report.save_table(summary, "summary_statistics")

    # 2. Shape distributions
    with report.figure("shape_distributions", figsize=(15, 4)) as fig:
        axes = fig.subplots(1, 3)

        sns.histplot(df[VolumeCol.WIDTH], bins=30, kde=False, ax=axes[0])
        axes[0].set_xlabel("Width (voxels)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Width Distribution")

        sns.histplot(df[VolumeCol.HEIGHT], bins=30, kde=False, ax=axes[1])
        axes[1].set_xlabel("Height (voxels)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Height Distribution")

        sns.histplot(df[VolumeCol.N_SLICES], bins=30, kde=False, ax=axes[2])
        axes[2].set_xlabel("Number of Slices")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Slice Count Distribution")

    # 3. Spacing distributions
    with report.figure("spacing_distributions", figsize=(15, 4)) as fig:
        axes = fig.subplots(1, 3)

        sns.histplot(df[VolumeCol.SPACING_X], bins=30, kde=False, ax=axes[0])
        axes[0].set_xlabel("Spacing X (mm)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("In-Plane Spacing X Distribution")

        sns.histplot(df[VolumeCol.SPACING_Y], bins=30, kde=False, ax=axes[1])
        axes[1].set_xlabel("Spacing Y (mm)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("In-Plane Spacing Y Distribution")

        sns.histplot(df[VolumeCol.SPACING_Z], bins=30, kde=False, ax=axes[2])
        axes[2].set_xlabel("Spacing Z (mm)")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Slice Thickness Distribution")

    # 4. Physical size distributions
    with report.figure("physical_size_distributions", figsize=(15, 4)) as fig:
        axes = fig.subplots(1, 3)

        sns.histplot(df[VolumeCol.PHYSICAL_WIDTH], bins=30, kde=False, ax=axes[0])
        axes[0].set_xlabel("Physical Width (mm)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Physical Width Distribution")

        sns.histplot(df[VolumeCol.PHYSICAL_HEIGHT], bins=30, kde=False, ax=axes[1])
        axes[1].set_xlabel("Physical Height (mm)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Physical Height Distribution")

        sns.histplot(df[VolumeCol.PHYSICAL_DEPTH], bins=30, kde=False, ax=axes[2])
        axes[2].set_xlabel("Physical Depth (mm)")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Physical Depth Distribution")

    # 5. Outlier detection
    outlier_data = []
    for col in [
        VolumeCol.WIDTH,
        VolumeCol.HEIGHT,
        VolumeCol.N_SLICES,
        VolumeCol.SPACING_X,
        VolumeCol.SPACING_Y,
        VolumeCol.SPACING_Z,
    ]:
        mean = df[col].mean()
        std = df[col].std()

        n_2sigma = ((df[col] < mean - 2 * std) | (df[col] > mean + 2 * std)).sum()
        n_3sigma = ((df[col] < mean - 3 * std) | (df[col] > mean + 3 * std)).sum()

        outlier_data.append(
            {
                "property": col,
                "mean": mean,
                "std": std,
                "n_outliers_2sigma": n_2sigma,
                "n_outliers_3sigma": n_3sigma,
                "pct_2sigma": f"{100 * n_2sigma / df.height:.1f}%",
                "pct_3sigma": f"{100 * n_3sigma / df.height:.1f}%",
            }
        )

        report.log_stat(f"{col}_outliers_2sigma", n_2sigma)
        report.log_stat(f"{col}_outliers_3sigma", n_3sigma)

    report.save_table(pl.DataFrame(outlier_data), "outlier_summary")
