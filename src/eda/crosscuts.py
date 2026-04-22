import polars as pl
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from src.data.loader import load_metadata
from src.data.schemas import Col
from src.eda.report import EDAReport
from src.eda.stats import chi2_result, mann_whitney_result

sns.set_theme(style="whitegrid", palette="muted")


def run(df, report_name: str) -> None:
    stat_results: list[tuple[str, dict]] = []

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

        a = df.filter(pl.col(Col.SEX) == "Female")[Col.AGE].drop_nulls()
        b = df.filter(pl.col(Col.SEX) == "Male")[Col.AGE].drop_nulls()
        res = mann_whitney_result(a, b)
        report.log_stat("mann_whitney_age_by_sex", res)
        stat_results.append(("age_by_sex", res))

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

        a = df.filter(pl.col(Col.RACE) == "White")[Col.AGE].drop_nulls()
        b = df.filter(pl.col(Col.RACE) == "Black or African American")[Col.AGE].drop_nulls()
        res = mann_whitney_result(a, b)
        report.log_stat("mann_whitney_age_by_race", res)
        stat_results.append(("age_by_race", res))

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

        res = chi2_result(cross_pd)
        report.log_stat("chi2_sex_by_race", res)
        stat_results.append(("sex_by_race", res))

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

        res = chi2_result(cross_pd)
        report.log_stat("chi2_race_by_manufacturer", res)
        stat_results.append(("race_by_manufacturer", res))

        with report.figure("sex_by_manufacturer", figsize=(6, 3)) as fig:
            ax = fig.subplots()
            pivot = (
                df.group_by(Col.SEX, Col.MANUFACTURER)
                .len()
                .pivot(on=Col.MANUFACTURER, index=Col.SEX, values="len")
                .fill_null(0)
            )
            cross_pd = pivot.to_pandas().set_index("sex")
            sns.heatmap(cross_pd, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Manufacturer")
            ax.set_ylabel("Sex")
            ax.set_title("Sex by Manufacturer (Counts)")

        res = chi2_result(cross_pd)
        report.log_stat("chi2_sex_by_manufacturer", res)
        stat_results.append(("sex_by_manufacturer", res))

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

        res = chi2_result(cross_pd)
        report.log_stat("chi2_race_by_field_strength", res)
        stat_results.append(("race_by_field_strength", res))

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

        manufacturers = sorted(df[Col.MANUFACTURER].drop_nulls().unique().to_list())
        a = df.filter(pl.col(Col.MANUFACTURER) == manufacturers[0])[Col.AGE].drop_nulls()
        b = df.filter(pl.col(Col.MANUFACTURER) == manufacturers[1])[Col.AGE].drop_nulls()
        res = mann_whitney_result(a, b)
        report.log_stat("mann_whitney_age_by_manufacturer", res)
        stat_results.append(("age_by_manufacturer", res))

        with report.figure("age_by_field_strength", figsize=(6, 5)) as fig:
            ax = fig.subplots()
            sns.violinplot(
                data=df.select("field_strength", Col.AGE).drop_nulls(),
                x="field_strength",
                y=Col.AGE,
                ax=ax,
            )
            ax.set_xlabel("Field Strength")
            ax.set_ylabel("Age at Imaging")
            ax.set_title("Age Distribution by Field Strength")

        field_strengths = sorted(df["field_strength"].drop_nulls().unique().to_list())
        a = df.filter(pl.col("field_strength") == field_strengths[0])[Col.AGE].drop_nulls()
        b = df.filter(pl.col("field_strength") == field_strengths[1])[Col.AGE].drop_nulls()
        res = mann_whitney_result(a, b)
        report.log_stat("mann_whitney_age_by_field_strength", res)
        stat_results.append(("age_by_field_strength", res))

        # ------------------------------------------------------------------
        # BH-FDR correction across all omnibus p-values in this module
        # ------------------------------------------------------------------

        p_values = [r["p"] for _, r in stat_results]
        reject, p_adj, _, _ = multipletests(p_values, method="fdr_bh")

        summary_rows = []
        for (key, r), p_a, rej in zip(stat_results, p_adj, reject):
            effect_name = "r_rb" if "r_rb" in r else "cramers_v"
            summary_rows.append({
                "stat_key": key,
                "test": r["test"],
                "p_raw": r["p"],
                "p_adjusted": float(p_a),
                "reject_h0": bool(rej),
                "effect_size": r.get(effect_name),
                "effect_name": effect_name,
            })

        report.save_table(pl.DataFrame(summary_rows), "stats_summary")


if __name__ == "__main__":
    run(load_metadata(), "full/crosscuts")
