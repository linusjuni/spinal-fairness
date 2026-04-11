import seaborn as sns

from src.data.loader import load_metadata
from src.data.schemas import Col
from src.eda.report import EDAReport

sns.set_theme(style="whitegrid", palette="muted")


def run(df, report_name: str) -> None:
    with EDAReport(report_name) as report:
        ages = df[Col.AGE].drop_nulls()

        with report.figure("age_distribution", figsize=(10, 5)) as fig:
            ax = fig.subplots()
            sns.histplot(ages, bins=30, kde=False, ax=ax)
            ax.set_xlabel("Age at Imaging")
            ax.set_ylabel("Count")
            ax.set_title("Age Distribution")

        with report.figure("sex_distribution", figsize=(6, 5)) as fig:
            ax = fig.subplots()
            counts = df[Col.SEX].value_counts().sort("count", descending=True)
            sns.barplot(x=counts[Col.SEX], y=counts["count"], ax=ax)
            ax.set_xlabel("Sex")
            ax.set_ylabel("Count")
            ax.set_title("Sex Distribution")

        with report.figure("race_distribution", figsize=(10, 5)) as fig:
            ax = fig.subplots()
            counts = df[Col.RACE].value_counts().sort("count", descending=True)
            sns.barplot(x=counts[Col.RACE], y=counts["count"], ax=ax)
            ax.set_xlabel("Race")
            ax.set_ylabel("Count")
            ax.set_title("Race Distribution")
            ax.tick_params(axis="x", rotation=30)

        with report.figure("ethnicity_distribution", figsize=(8, 5)) as fig:
            ax = fig.subplots()
            counts = df[Col.ETHNICITY].value_counts().sort("count", descending=True)
            sns.barplot(x=counts[Col.ETHNICITY], y=counts["count"], ax=ax)
            ax.set_xlabel("Ethnicity")
            ax.set_ylabel("Count")
            ax.set_title("Ethnicity Distribution")
            ax.tick_params(axis="x", rotation=30)


if __name__ == "__main__":
    run(load_metadata(), "full/demographics")
