import polars as pl
import seaborn as sns

from src.data.loader import load_metadata
from src.data.volumes import load_volume_properties
from src.eda.report import EDAReport

sns.set_theme(style="whitegrid", palette="muted")

# Load data
volumes = load_volume_properties()
metadata = load_metadata()

# Merge on series_submitter_id
df = volumes.join(metadata, on="series_submitter_id", how="left")

with EDAReport("volumes") as report:
    # 1. Summary statistics
    report.log_stat("total_images", df.height)
    report.log_stat(
        "shape_stats",
        {
            "width": {
                "mean": df["width"].mean(),
                "std": df["width"].std(),
                "min": df["width"].min(),
                "max": df["width"].max(),
            },
            "height": {
                "mean": df["height"].mean(),
                "std": df["height"].std(),
                "min": df["height"].min(),
                "max": df["height"].max(),
            },
            "n_slices": {
                "mean": df["n_slices"].mean(),
                "std": df["n_slices"].std(),
                "min": df["n_slices"].min(),
                "max": df["n_slices"].max(),
            },
        },
    )
    report.log_stat(
        "spacing_stats",
        {
            "spacing_x": {
                "mean": df["spacing_x"].mean(),
                "std": df["spacing_x"].std(),
            },
            "spacing_y": {
                "mean": df["spacing_y"].mean(),
                "std": df["spacing_y"].std(),
            },
            "spacing_z": {
                "mean": df["spacing_z"].mean(),
                "std": df["spacing_z"].std(),
            },
        },
    )

    # Save summary table
    summary = df.select(
        [
            pl.col("width").mean().alias("mean_width"),
            pl.col("width").std().alias("std_width"),
            pl.col("width").min().alias("min_width"),
            pl.col("width").max().alias("max_width"),
            pl.col("height").mean().alias("mean_height"),
            pl.col("height").std().alias("std_height"),
            pl.col("height").min().alias("min_height"),
            pl.col("height").max().alias("max_height"),
            pl.col("n_slices").mean().alias("mean_n_slices"),
            pl.col("n_slices").std().alias("std_n_slices"),
            pl.col("n_slices").min().alias("min_n_slices"),
            pl.col("n_slices").max().alias("max_n_slices"),
            pl.col("spacing_x").mean().alias("mean_spacing_x"),
            pl.col("spacing_x").std().alias("std_spacing_x"),
            pl.col("spacing_y").mean().alias("mean_spacing_y"),
            pl.col("spacing_y").std().alias("std_spacing_y"),
            pl.col("spacing_z").mean().alias("mean_spacing_z"),
            pl.col("spacing_z").std().alias("std_spacing_z"),
        ]
    )
    report.save_table(summary, "summary_statistics")

    # 2. Shape distributions
    with report.figure("shape_distributions", figsize=(15, 4)) as fig:
        axes = fig.subplots(1, 3)

        sns.histplot(df["width"], bins=30, kde=True, ax=axes[0])
        axes[0].set_xlabel("Width (voxels)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Width Distribution")

        sns.histplot(df["height"], bins=30, kde=True, ax=axes[1])
        axes[1].set_xlabel("Height (voxels)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Height Distribution")

        sns.histplot(df["n_slices"], bins=30, kde=True, ax=axes[2])
        axes[2].set_xlabel("Number of Slices")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Slice Count Distribution")

    # 3. Spacing distributions
    with report.figure("spacing_distributions", figsize=(15, 4)) as fig:
        axes = fig.subplots(1, 3)

        sns.histplot(df["spacing_x"], bins=30, kde=True, ax=axes[0])
        axes[0].set_xlabel("Spacing X (mm)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("In-Plane Spacing X Distribution")

        sns.histplot(df["spacing_y"], bins=30, kde=True, ax=axes[1])
        axes[1].set_xlabel("Spacing Y (mm)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("In-Plane Spacing Y Distribution")

        sns.histplot(df["spacing_z"], bins=30, kde=True, ax=axes[2])
        axes[2].set_xlabel("Spacing Z (mm)")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Slice Thickness Distribution")

    # 4. Physical size distributions
    with report.figure("physical_size_distributions", figsize=(15, 4)) as fig:
        axes = fig.subplots(1, 3)

        sns.histplot(df["physical_width"], bins=30, kde=True, ax=axes[0])
        axes[0].set_xlabel("Physical Width (mm)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Physical Width Distribution")

        sns.histplot(df["physical_height"], bins=30, kde=True, ax=axes[1])
        axes[1].set_xlabel("Physical Height (mm)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Physical Height Distribution")

        sns.histplot(df["physical_depth"], bins=30, kde=True, ax=axes[2])
        axes[2].set_xlabel("Physical Depth (mm)")
        axes[2].set_ylabel("Count")
        axes[2].set_title("Physical Depth Distribution")

    # 5. Outlier detection
    outlier_data = []
    for col in ["width", "height", "n_slices", "spacing_x", "spacing_y", "spacing_z"]:
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
