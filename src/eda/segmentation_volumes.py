import numpy as np
import polars as pl
import seaborn as sns

from src.data.loader import load_metadata
from src.data.schemas import Col, SegmentationVolumeCol
from src.data.segmentation_volumes import load_segmentation_volumes
from src.eda.report import EDAReport

sns.set_theme(style="whitegrid", palette="muted")

AGE_BIN_ORDER = ["<40", "40-60", "60+"]
RACE_ORDER = ["White", "Black or African American"]
SEX_ORDER = ["Female", "Male"]


def run(df, report_name: str) -> None:
    df = df.with_columns(
        pl.when(pl.col(Col.AGE).is_null())
        .then(pl.lit("60+"))
        .when(pl.col(Col.AGE) < 40)
        .then(pl.lit("<40"))
        .when(pl.col(Col.AGE) < 60)
        .then(pl.lit("40-60"))
        .otherwise(pl.lit("60+"))
        .alias("age_bin")
    )

    df_race = df.filter(pl.col(Col.RACE).is_in(RACE_ORDER))

    with EDAReport(report_name) as report:

        # ------------------------------------------------------------------
        # 1. Distributions
        # ------------------------------------------------------------------

        with report.figure("dist_volume_mm3", figsize=(12, 4)) as fig:
            axes = fig.subplots(1, 2)
            sns.histplot(df[SegmentationVolumeCol.VOLUME_MM3_VERTEBRAL_BODY], bins=40, ax=axes[0])
            axes[0].set_xlabel("Volume (mm³)")
            axes[0].set_title("Vertebral Body Volume (mm³)")
            sns.histplot(df[SegmentationVolumeCol.VOLUME_MM3_DISC], bins=40, ax=axes[1])
            axes[1].set_xlabel("Volume (mm³)")
            axes[1].set_title("Intervertebral Disc Volume (mm³)")

        with report.figure("dist_voxel_counts", figsize=(12, 4)) as fig:
            axes = fig.subplots(1, 2)
            sns.histplot(df[SegmentationVolumeCol.N_VOXELS_VERTEBRAL_BODY], bins=40, ax=axes[0])
            axes[0].set_xlabel("Voxel count")
            axes[0].set_title("Vertebral Body Voxel Count")
            axes[0].tick_params(axis="x", rotation=45)
            sns.histplot(df[SegmentationVolumeCol.N_VOXELS_DISC], bins=40, ax=axes[1])
            axes[1].set_xlabel("Voxel count")
            axes[1].set_title("Intervertebral Disc Voxel Count")
            axes[1].tick_params(axis="x", rotation=45)

        with report.figure("dist_components", figsize=(12, 4)) as fig:
            axes = fig.subplots(1, 2)
            vb = df[SegmentationVolumeCol.N_COMPONENTS_VERTEBRAL_BODY]
            disc = df[SegmentationVolumeCol.N_COMPONENTS_DISC]
            sns.histplot(vb, bins=np.arange(vb.min(), vb.max() + 2) - 0.5, ax=axes[0])
            axes[0].set_xlabel("Component count")
            axes[0].set_title("Vertebral Body Components")
            sns.histplot(disc, bins=np.arange(disc.min(), disc.max() + 2) - 0.5, ax=axes[1])
            axes[1].set_xlabel("Component count")
            axes[1].set_title("Intervertebral Disc Components")

        outliers = df.filter(
            (pl.col(SegmentationVolumeCol.N_COMPONENTS_VERTEBRAL_BODY) > 15)
            | (pl.col(SegmentationVolumeCol.N_COMPONENTS_DISC) > 15)
        ).select(
            Col.SERIES_SUBMITTER_ID,
            Col.FILENAME,
            SegmentationVolumeCol.N_COMPONENTS_VERTEBRAL_BODY,
            SegmentationVolumeCol.N_COMPONENTS_DISC,
            Col.SEX,
            Col.RACE,
            Col.AGE,
            Col.MANUFACTURER,
            Col.FIELD_STRENGTH,
        )
        report.save_table(outliers, "high_component_outliers")
        report.log_stat("n_high_component_outliers", outliers.height)

        # ------------------------------------------------------------------
        # 2. Volume by demographics (anatomy / confounder check)
        # ------------------------------------------------------------------

        for label, col in [
            ("Vertebral Body", SegmentationVolumeCol.VOLUME_MM3_VERTEBRAL_BODY),
            ("Disc", SegmentationVolumeCol.VOLUME_MM3_DISC),
        ]:
            with report.figure(f"vol_by_sex_{col}", figsize=(6, 5)) as fig:
                ax = fig.subplots()
                sns.violinplot(
                    data=df.select(Col.SEX, col).to_pandas(),
                    x=Col.SEX,
                    y=col,
                    order=SEX_ORDER,
                    ax=ax,
                )
                ax.set_xlabel("Sex")
                ax.set_ylabel("Volume (mm³)")
                ax.set_title(f"{label} Volume by Sex")

            with report.figure(f"vol_by_race_{col}", figsize=(6, 5)) as fig:
                ax = fig.subplots()
                sns.violinplot(
                    data=df_race.select(Col.RACE, col).to_pandas(),
                    x=Col.RACE,
                    y=col,
                    order=RACE_ORDER,
                    ax=ax,
                )
                ax.set_xlabel("Race")
                ax.set_ylabel("Volume (mm³)")
                ax.set_title(f"{label} Volume by Race")
                ax.tick_params(axis="x", rotation=15)

            with report.figure(f"vol_by_age_{col}", figsize=(7, 5)) as fig:
                ax = fig.subplots()
                sns.violinplot(
                    data=df.select("age_bin", col).to_pandas(),
                    x="age_bin",
                    y=col,
                    order=AGE_BIN_ORDER,
                    ax=ax,
                )
                ax.set_xlabel("Age Bin")
                ax.set_ylabel("Volume (mm³)")
                ax.set_title(f"{label} Volume by Age Bin")

        # ------------------------------------------------------------------
        # 3. Voxel counts by scanner (resolution dependency check)
        # ------------------------------------------------------------------

        for label, col in [
            ("Vertebral Body", SegmentationVolumeCol.N_VOXELS_VERTEBRAL_BODY),
            ("Disc", SegmentationVolumeCol.N_VOXELS_DISC),
        ]:
            with report.figure(f"voxels_by_manufacturer_{col}", figsize=(6, 5)) as fig:
                ax = fig.subplots()
                sns.violinplot(
                    data=df.select(Col.MANUFACTURER, col).to_pandas(),
                    x=Col.MANUFACTURER,
                    y=col,
                    ax=ax,
                )
                ax.set_xlabel("Manufacturer")
                ax.set_ylabel("Voxel count")
                ax.set_title(f"{label} Voxel Count by Manufacturer")
                ax.tick_params(axis="x", rotation=15)

        for label, col in [
            ("Vertebral Body", SegmentationVolumeCol.VOLUME_MM3_VERTEBRAL_BODY),
            ("Disc", SegmentationVolumeCol.VOLUME_MM3_DISC),
        ]:
            with report.figure(f"vol_mm3_by_manufacturer_{col}", figsize=(6, 5)) as fig:
                ax = fig.subplots()
                sns.violinplot(
                    data=df.select(Col.MANUFACTURER, col).to_pandas(),
                    x=Col.MANUFACTURER,
                    y=col,
                    ax=ax,
                )
                ax.set_xlabel("Manufacturer")
                ax.set_ylabel("Volume (mm³)")
                ax.set_title(f"{label} Volume (mm³) by Manufacturer")
                ax.tick_params(axis="x", rotation=15)

        for col in [
            SegmentationVolumeCol.VOLUME_MM3_VERTEBRAL_BODY,
            SegmentationVolumeCol.VOLUME_MM3_DISC,
        ]:
            stats = (
                df.group_by(Col.MANUFACTURER)
                .agg(
                    pl.col(col).mean().alias("mean"),
                    pl.col(col).std().alias("std"),
                    pl.col(col).median().alias("median"),
                )
                .sort(Col.MANUFACTURER)
            )
            report.log_stat(f"{col}_by_manufacturer", stats.to_dicts())

        # ------------------------------------------------------------------
        # 4. Component counts by demographics (annotation completeness check)
        # ------------------------------------------------------------------

        for label, col in [
            ("Vertebral Body", SegmentationVolumeCol.N_COMPONENTS_VERTEBRAL_BODY),
            ("Disc", SegmentationVolumeCol.N_COMPONENTS_DISC),
        ]:
            with report.figure(f"comp_by_sex_{col}", figsize=(6, 5)) as fig:
                ax = fig.subplots()
                sns.violinplot(
                    data=df.select(Col.SEX, col).to_pandas(),
                    x=Col.SEX,
                    y=col,
                    order=SEX_ORDER,
                    ax=ax,
                )
                ax.set_xlabel("Sex")
                ax.set_ylabel("Component count")
                ax.set_title(f"{label} Components by Sex")

            with report.figure(f"comp_by_race_{col}", figsize=(6, 5)) as fig:
                ax = fig.subplots()
                sns.violinplot(
                    data=df_race.select(Col.RACE, col).to_pandas(),
                    x=Col.RACE,
                    y=col,
                    order=RACE_ORDER,
                    ax=ax,
                )
                ax.set_xlabel("Race")
                ax.set_ylabel("Component count")
                ax.set_title(f"{label} Components by Race")
                ax.tick_params(axis="x", rotation=15)

            with report.figure(f"comp_by_age_{col}", figsize=(7, 5)) as fig:
                ax = fig.subplots()
                sns.violinplot(
                    data=df.select("age_bin", col).to_pandas(),
                    x="age_bin",
                    y=col,
                    order=AGE_BIN_ORDER,
                    ax=ax,
                )
                ax.set_xlabel("Age Bin")
                ax.set_ylabel("Component count")
                ax.set_title(f"{label} Components by Age Bin")


if __name__ == "__main__":
    seg = load_segmentation_volumes()
    metadata = load_metadata()
    df = seg.join(metadata, on=Col.SERIES_SUBMITTER_ID, how="left")
    run(df, "full/segmentation_volumes")
