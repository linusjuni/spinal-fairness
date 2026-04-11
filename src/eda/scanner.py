import seaborn as sns

from src.data.loader import load_metadata
from src.data.schemas import Col
from src.eda.report import EDAReport

sns.set_theme(style="whitegrid", palette="muted")


def run(df, report_name: str) -> None:
    with EDAReport(report_name) as report:
        with report.figure("manufacturer", figsize=(8, 5)) as fig:
            ax = fig.subplots()
            counts = df[Col.MANUFACTURER].value_counts().sort("count", descending=True)
            sns.barplot(x=counts[Col.MANUFACTURER], y=counts["count"], ax=ax)
            ax.set_xlabel("Manufacturer")
            ax.set_ylabel("Count")
            ax.set_title("Manufacturer Distribution")

        with report.figure("manufacturer_model", figsize=(10, 5)) as fig:
            ax = fig.subplots()
            counts = df[Col.MANUFACTURER_MODEL].value_counts().sort("count", descending=True)
            sns.barplot(x=counts[Col.MANUFACTURER_MODEL], y=counts["count"], ax=ax)
            ax.set_xlabel("Model")
            ax.set_ylabel("Count")
            ax.set_title("Scanner Model Distribution")
            ax.tick_params(axis="x", rotation=30)

        with report.figure("field_strength", figsize=(6, 5)) as fig:
            ax = fig.subplots()
            counts = df[Col.FIELD_STRENGTH].value_counts().sort(Col.FIELD_STRENGTH)
            labels = [f"{v}T" for v in counts[Col.FIELD_STRENGTH].to_list()]
            sns.barplot(x=labels, y=counts["count"], ax=ax)
            ax.set_xlabel("Field Strength")
            ax.set_ylabel("Count")
            ax.set_title("Magnetic Field Strength Distribution")

        with report.figure("slice_thickness", figsize=(6, 5)) as fig:
            ax = fig.subplots()
            counts = df[Col.SLICE_THICKNESS].value_counts().sort(Col.SLICE_THICKNESS)
            labels = [f"{v} mm" for v in counts[Col.SLICE_THICKNESS].to_list()]
            sns.barplot(x=labels, y=counts["count"], ax=ax)
            ax.set_xlabel("Slice Thickness")
            ax.set_ylabel("Count")
            ax.set_title("Slice Thickness Distribution")

        with report.figure("pixel_spacing", figsize=(10, 5)) as fig:
            ax = fig.subplots()
            sns.histplot(df[Col.PIXEL_SPACING], bins=25, ax=ax)
            ax.set_xlabel("Pixel Spacing (mm)")
            ax.set_ylabel("Count")
            ax.set_title("Pixel Spacing Distribution")

        with report.figure("echo_time", figsize=(10, 5)) as fig:
            ax = fig.subplots()
            sns.histplot(df[Col.ECHO_TIME], bins=30, ax=ax)
            ax.set_xlabel("Echo Time (ms)")
            ax.set_ylabel("Count")
            ax.set_title("Echo Time Distribution")

        with report.figure("repetition_time", figsize=(10, 5)) as fig:
            ax = fig.subplots()
            sns.histplot(df[Col.REPETITION_TIME], bins=30, ax=ax)
            ax.set_xlabel("Repetition Time (ms)")
            ax.set_ylabel("Count")
            ax.set_title("Repetition Time Distribution")


if __name__ == "__main__":
    run(load_metadata(), "full/scanner")
