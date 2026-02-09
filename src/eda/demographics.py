import seaborn as sns

from src.data.loader import load_metadata
from src.eda.report import EDAReport

sns.set_theme(style="whitegrid", palette="muted")

df = load_metadata()

with EDAReport("demographics") as report:
    # Age distribution
    ages = df["age_at_imaging"].drop_nulls()

    # Age distribution
    with report.figure("age_distribution", figsize=(10, 5)) as fig:
        ax = fig.subplots()
        sns.histplot(ages, bins=30, kde=False, ax=ax)
        ax.set_xlabel("Age at Imaging")
        ax.set_ylabel("Count")
        ax.set_title("Age Distribution")

    # Sex distribution
    with report.figure("sex_distribution", figsize=(6, 5)) as fig:
        ax = fig.subplots()
        counts = df["sex"].value_counts().sort("count", descending=True)
        sns.barplot(x=counts["sex"], y=counts["count"], ax=ax)
        ax.set_xlabel("Sex")
        ax.set_ylabel("Count")
        ax.set_title("Sex Distribution")

    # Race distribution
    with report.figure("race_distribution", figsize=(10, 5)) as fig:
        ax = fig.subplots()
        counts = df["race"].value_counts().sort("count", descending=True)
        sns.barplot(x=counts["race"], y=counts["count"], ax=ax)
        ax.set_xlabel("Race")
        ax.set_ylabel("Count")
        ax.set_title("Race Distribution")
        ax.tick_params(axis="x", rotation=30)

    # Ethnicity distribution
    with report.figure("ethnicity_distribution", figsize=(8, 5)) as fig:
        ax = fig.subplots()
        counts = df["ethnicity"].value_counts().sort("count", descending=True)
        sns.barplot(x=counts["ethnicity"], y=counts["count"], ax=ax)
        ax.set_xlabel("Ethnicity")
        ax.set_ylabel("Count")
        ax.set_title("Ethnicity Distribution")
        ax.tick_params(axis="x", rotation=30)
