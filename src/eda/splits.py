import polars as pl
import seaborn as sns

from src.data.loader import load_metadata
from src.data.schemas import Col
from src.data.splits import apply_splits
from src.eda.report import EDAReport

sns.set_theme(style="whitegrid", palette="muted")

SPLIT_ORDER = ["train", "val", "test"]

VERSION = "split_v1"

df = apply_splits(load_metadata(), version=VERSION)

with EDAReport(f"splits/{VERSION}") as report:
    # 1. Overall split counts
    with report.figure("split_counts", figsize=(6, 4)) as fig:
        ax = fig.subplots()
        counts = (
            df.group_by("split").len().sort(pl.col("split").cast(pl.Enum(SPLIT_ORDER)))
        )
        sns.barplot(x=counts["split"], y=counts["len"], order=SPLIT_ORDER, ax=ax)
        ax.set_xlabel("Split")
        ax.set_ylabel("Count")
        ax.set_title("Exam Count per Split")
        for bar, val in zip(ax.patches, counts["len"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                str(val),
                ha="center",
                va="bottom",
                fontsize=10,
            )

    # 2. Race distribution per split (primary groups only)
    with report.figure("race_per_split", figsize=(8, 4)) as fig:
        ax = fig.subplots()
        counts = df.group_by("split", "race_bin").len().to_pandas()
        sns.barplot(
            data=counts,
            x="split",
            y="len",
            hue="race_bin",
            order=SPLIT_ORDER,
            hue_order=["White", "Black", "Other"],
            ax=ax,
        )
        ax.set_xlabel("Split")
        ax.set_ylabel("Count")
        ax.set_title("Race Count per Split")
        ax.legend(title="Race")

    # 3. Race % per split (normalised — checks proportions are consistent)
    with report.figure("race_pct_per_split", figsize=(8, 4)) as fig:
        ax = fig.subplots()
        pct = (
            df.group_by("split", "race_bin")
            .len()
            .with_columns(
                (pl.col("len") / pl.col("len").sum().over("split") * 100).alias("pct")
            )
            .to_pandas()
        )
        sns.barplot(
            data=pct,
            x="split",
            y="pct",
            hue="race_bin",
            order=SPLIT_ORDER,
            hue_order=["White", "Black", "Other"],
            ax=ax,
        )
        ax.set_xlabel("Split")
        ax.set_ylabel("% within split")
        ax.set_title("Race Proportion per Split")
        ax.legend(title="Race")

    # 4. Age bin proportion per split (% within split, not raw counts)
    with report.figure("age_per_split", figsize=(8, 4)) as fig:
        ax = fig.subplots()
        pct = (
            df.group_by("split", "age_bin")
            .len()
            .with_columns(
                (pl.col("len") / pl.col("len").sum().over("split") * 100).alias("pct")
            )
            .to_pandas()
        )
        sns.barplot(
            data=pct,
            x="split",
            y="pct",
            hue="age_bin",
            order=SPLIT_ORDER,
            hue_order=["<40", "40-60", "60+"],
            ax=ax,
        )
        ax.set_xlabel("Split")
        ax.set_ylabel("% within split")
        ax.set_title("Age Bin Proportion per Split")
        ax.legend(title="Age Bin")

    # 5. Race × age heatmap — one panel per split (key stratification check)
    # Shared colour scale so panels are visually comparable (normalised to %).
    with report.figure("race_age_heatmap", figsize=(14, 4)) as fig:
        axes = fig.subplots(1, 3)
        ROW_ORDER = ["White", "Black"]
        COL_ORDER = ["<40", "40-60", "60+"]

        # Build all pivots first so we can compute a shared vmax
        pivots = {}
        for split in SPLIT_ORDER:
            pivot = (
                df.filter(
                    (pl.col("split") == split) & pl.col("race_bin").is_in(ROW_ORDER)
                )
                .group_by("race_bin", "age_bin")
                .len()
                .with_columns((pl.col("len") / pl.col("len").sum() * 100).alias("pct"))
                .pivot(on="age_bin", index="race_bin", values="pct")
                .fill_null(0)
                .to_pandas()
                .set_index("race_bin")
            )
            for col in COL_ORDER:
                if col not in pivot.columns:
                    pivot[col] = 0.0
            pivots[split] = pivot[COL_ORDER].loc[ROW_ORDER]

        vmax = max(p.values.max() for p in pivots.values())

        for ax, split in zip(axes, SPLIT_ORDER):
            sns.heatmap(
                pivots[split],
                annot=True,
                fmt=".1f",
                cmap="Blues",
                vmin=0,
                vmax=vmax,
                ax=ax,
            )
            ax.set_title(split.capitalize())
            ax.set_xlabel("Age Bin")
            ax.set_ylabel("Race")
        fig.suptitle("Race × Age % per Split (shared colour scale)", y=1.02)

    # 6. Sex distribution per split (post-hoc)
    with report.figure("sex_per_split", figsize=(8, 4)) as fig:
        ax = fig.subplots()
        pct = (
            df.group_by("split", Col.SEX)
            .len()
            .with_columns(
                (pl.col("len") / pl.col("len").sum().over("split") * 100).alias("pct")
            )
            .to_pandas()
        )
        sns.barplot(
            data=pct,
            x="split",
            y="pct",
            hue=Col.SEX,
            order=SPLIT_ORDER,
            hue_order=["Male", "Female"],
            ax=ax,
        )
        ax.set_xlabel("Split")
        ax.set_ylabel("% within split")
        ax.set_title("Sex Proportion per Split")
        ax.legend(title="Sex")
        ax.axhline(50, color="grey", linestyle="--", linewidth=0.8)
