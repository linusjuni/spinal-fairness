import polars as pl
import seaborn as sns

from src.data.loader import load_metadata
from src.data.schemas import Col
from src.eda.report import EDAReport

sns.set_theme(style="whitegrid", palette="muted")


def run(df, report_name: str) -> None:
    df = df.with_columns(
        (pl.col(Col.FIELD_STRENGTH).cast(pl.String) + "T").alias("field_strength")
    )

    with EDAReport(report_name) as report:
        with report.figure("age_by_sex", figsize=(8, 5)) as fig:
            ax = fig.subplots()
            sns.violinplot(
                data=df.select(Col.SEX, Col.AGE).drop_nulls(),
                x=Col.SEX,
                y=Col.AGE,
                ax=ax,
            )
            ax.set_xlabel("Sex")
            ax.set_ylabel("Age at Imaging")
            ax.set_title("Age Distribution by Sex")

        with report.figure("age_by_race", figsize=(12, 5)) as fig:
            ax = fig.subplots()
            sns.violinplot(
                data=df.select(Col.RACE, Col.AGE).drop_nulls(),
                x=Col.RACE,
                y=Col.AGE,
                ax=ax,
            )
            ax.set_xlabel("Race")
            ax.set_ylabel("Age at Imaging")
            ax.set_title("Age Distribution by Race")
            ax.tick_params(axis="x", rotation=30)

        with report.figure("sex_by_race", figsize=(10, 4)) as fig:
            ax = fig.subplots()
            pivot = (
                df.group_by(Col.SEX, Col.RACE)
                .len()
                .pivot(on=Col.RACE, index=Col.SEX, values="len")
                .fill_null(0)
            )
            cross_pd = pivot.to_pandas().set_index("sex")
            race_order = cross_pd.sum().sort_values(ascending=False).index.tolist()
            cross_pd = cross_pd[race_order]
            sns.heatmap(cross_pd, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Race")
            ax.set_ylabel("Sex")
            ax.set_title("Sex by Race (Counts)")

        with report.figure("race_by_manufacturer", figsize=(8, 4)) as fig:
            ax = fig.subplots()
            pivot = (
                df.group_by(Col.RACE, Col.MANUFACTURER)
                .len()
                .pivot(on=Col.MANUFACTURER, index=Col.RACE, values="len")
                .fill_null(0)
            )
            cross_pd = pivot.to_pandas().set_index("race")
            race_order = cross_pd.sum(axis=1).sort_values(ascending=False).index.tolist()
            cross_pd = cross_pd.loc[race_order]
            sns.heatmap(cross_pd, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Manufacturer")
            ax.set_ylabel("Race")
            ax.set_title("Race by Manufacturer (Counts)")

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

        with report.figure("age_by_manufacturer", figsize=(8, 5)) as fig:
            ax = fig.subplots()
            sns.violinplot(
                data=df.select(Col.MANUFACTURER, Col.AGE).drop_nulls(),
                x=Col.MANUFACTURER,
                y=Col.AGE,
                ax=ax,
            )
            ax.set_xlabel("Manufacturer")
            ax.set_ylabel("Age at Imaging")
            ax.set_title("Age Distribution by Manufacturer")

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


if __name__ == "__main__":
    run(load_metadata(), "full/crosscuts")
