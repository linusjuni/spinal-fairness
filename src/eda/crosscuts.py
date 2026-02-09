"""EDA: demographic cross-cuts (age x sex, age x race, sex x race)."""

import seaborn as sns

from src.data.loader import load_metadata
from src.eda.report import EDAReport

sns.set_theme(style="whitegrid", palette="muted")

df = load_metadata()
df_pd = df.select("age_at_imaging", "sex", "race").drop_nulls().to_pandas()

with EDAReport("crosscuts") as report:
    # Age by sex
    with report.figure("age_by_sex", figsize=(8, 5)) as fig:
        ax = fig.subplots()
        sns.violinplot(data=df_pd, x="sex", y="age_at_imaging", ax=ax)
        ax.set_xlabel("Sex")
        ax.set_ylabel("Age at Imaging")
        ax.set_title("Age Distribution by Sex")

    # Age by race
    with report.figure("age_by_race", figsize=(12, 5)) as fig:
        ax = fig.subplots()
        sns.violinplot(data=df_pd, x="race", y="age_at_imaging", ax=ax)
        ax.set_xlabel("Race")
        ax.set_ylabel("Age at Imaging")
        ax.set_title("Age Distribution by Race")
        ax.tick_params(axis="x", rotation=30)

    # Sex by race (heatmap of counts)
    with report.figure("sex_by_race", figsize=(10, 4)) as fig:
        ax = fig.subplots()
        cross_pd = (
            df.group_by("sex", "race").len()
            .pivot(on="race", index="sex", values="len")
            .fill_null(0)
            .to_pandas()
            .set_index("sex")
        )
        race_order = cross_pd.sum().sort_values(ascending=False).index.tolist()
        cross_pd = cross_pd[race_order]
        sns.heatmap(cross_pd, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Race")
        ax.set_ylabel("Sex")
        ax.set_title("Sex by Race (Counts)")
