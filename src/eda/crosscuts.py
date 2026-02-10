import polars as pl
import seaborn as sns

from src.data.loader import load_metadata
from src.eda.report import EDAReport

sns.set_theme(style="whitegrid", palette="muted")

df = load_metadata()

# Add a string "field_strength" label for plotting (e.g. "3.0T" instead of 3.0)
df = df.with_columns(
    (pl.col("magnetic_field_strength").cast(pl.String) + "T").alias("field_strength")
)

with EDAReport("crosscuts") as report:
    # Age by sex
    with report.figure("age_by_sex", figsize=(8, 5)) as fig:
        ax = fig.subplots()
        sns.violinplot(
            data=df.select("sex", "age_at_imaging").drop_nulls(),
            x="sex",
            y="age_at_imaging",
            ax=ax,
        )
        ax.set_xlabel("Sex")
        ax.set_ylabel("Age at Imaging")
        ax.set_title("Age Distribution by Sex")

    # Age by race
    with report.figure("age_by_race", figsize=(12, 5)) as fig:
        ax = fig.subplots()
        sns.violinplot(
            data=df.select("race", "age_at_imaging").drop_nulls(),
            x="race",
            y="age_at_imaging",
            ax=ax,
        )
        ax.set_xlabel("Race")
        ax.set_ylabel("Age at Imaging")
        ax.set_title("Age Distribution by Race")
        ax.tick_params(axis="x", rotation=30)

    # Sex by race
    with report.figure("sex_by_race", figsize=(10, 4)) as fig:
        ax = fig.subplots()
        pivot = (
            df.group_by("sex", "race")
            .len()
            .pivot(on="race", index="sex", values="len")
            .fill_null(0)
        )
        cross_pd = pivot.to_pandas().set_index("sex")
        race_order = cross_pd.sum().sort_values(ascending=False).index.tolist()
        cross_pd = cross_pd[race_order]
        sns.heatmap(cross_pd, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Race")
        ax.set_ylabel("Sex")
        ax.set_title("Sex by Race (Counts)")

    # Race by manufacturer
    with report.figure("race_by_manufacturer", figsize=(8, 4)) as fig:
        ax = fig.subplots()
        pivot = (
            df.group_by("race", "manufacturer")
            .len()
            .pivot(on="manufacturer", index="race", values="len")
            .fill_null(0)
        )
        cross_pd = pivot.to_pandas().set_index("race")
        race_order = cross_pd.sum(axis=1).sort_values(ascending=False).index.tolist()
        cross_pd = cross_pd.loc[race_order]
        sns.heatmap(cross_pd, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Manufacturer")
        ax.set_ylabel("Race")
        ax.set_title("Race by Manufacturer (Counts)")

    # Race by field strength
    with report.figure("race_by_field_strength", figsize=(6, 4)) as fig:
        ax = fig.subplots()
        pivot = (
            df.group_by("race", "field_strength")
            .len()
            .pivot(on="field_strength", index="race", values="len")
            .fill_null(0)
        )
        cross_pd = pivot.to_pandas().set_index("race")
        race_order = cross_pd.sum(axis=1).sort_values(ascending=False).index.tolist()
        cross_pd = cross_pd.loc[race_order]
        sns.heatmap(cross_pd, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Field Strength")
        ax.set_ylabel("Race")
        ax.set_title("Race by Field Strength (Counts)")

    # Age by manufacturer
    with report.figure("age_by_manufacturer", figsize=(8, 5)) as fig:
        ax = fig.subplots()
        sns.violinplot(
            data=df.select("manufacturer", "age_at_imaging").drop_nulls(),
            x="manufacturer",
            y="age_at_imaging",
            ax=ax,
        )
        ax.set_xlabel("Manufacturer")
        ax.set_ylabel("Age at Imaging")
        ax.set_title("Age Distribution by Manufacturer")

    # Age by field strength
    with report.figure("age_by_field_strength", figsize=(6, 5)) as fig:
        ax = fig.subplots()
        sns.violinplot(
            data=df.select("field_strength", "age_at_imaging").drop_nulls(),
            x="field_strength",
            y="age_at_imaging",
            ax=ax,
        )
        ax.set_xlabel("Field Strength")
        ax.set_ylabel("Age at Imaging")
        ax.set_title("Age Distribution by Field Strength")
