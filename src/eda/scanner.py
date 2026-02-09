import seaborn as sns

from src.data.loader import load_metadata
from src.eda.report import EDAReport

sns.set_theme(style="whitegrid", palette="muted")

df = load_metadata()

with EDAReport("scanner") as report:
    # Manufacturer
    with report.figure("manufacturer", figsize=(8, 5)) as fig:
        ax = fig.subplots()
        counts = df["manufacturer"].value_counts().sort("count", descending=True)
        sns.barplot(x=counts["manufacturer"], y=counts["count"], ax=ax)
        ax.set_xlabel("Manufacturer")
        ax.set_ylabel("Count")
        ax.set_title("Manufacturer Distribution")

    # Manufacturer model
    with report.figure("manufacturer_model", figsize=(10, 5)) as fig:
        ax = fig.subplots()
        counts = (
            df["manufacturer_model_name"].value_counts().sort("count", descending=True)
        )
        sns.barplot(x=counts["manufacturer_model_name"], y=counts["count"], ax=ax)
        ax.set_xlabel("Model")
        ax.set_ylabel("Count")
        ax.set_title("Scanner Model Distribution")
        ax.tick_params(axis="x", rotation=30)

    # Field strength
    with report.figure("field_strength", figsize=(6, 5)) as fig:
        ax = fig.subplots()
        counts = (
            df["magnetic_field_strength"].value_counts().sort("magnetic_field_strength")
        )
        labels = [f"{v}T" for v in counts["magnetic_field_strength"].to_list()]
        sns.barplot(x=labels, y=counts["count"], ax=ax)
        ax.set_xlabel("Field Strength")
        ax.set_ylabel("Count")
        ax.set_title("Magnetic Field Strength Distribution")

    # Slice thickness
    with report.figure("slice_thickness", figsize=(6, 5)) as fig:
        ax = fig.subplots()
        counts = df["slice_thickness"].value_counts().sort("slice_thickness")
        labels = [f"{v} mm" for v in counts["slice_thickness"].to_list()]
        sns.barplot(x=labels, y=counts["count"], ax=ax)
        ax.set_xlabel("Slice Thickness")
        ax.set_ylabel("Count")
        ax.set_title("Slice Thickness Distribution")

    # Pixel spacing
    with report.figure("pixel_spacing", figsize=(10, 5)) as fig:
        ax = fig.subplots()
        sns.histplot(df["pixel_spacing"], bins=25, ax=ax)
        ax.set_xlabel("Pixel Spacing (mm)")
        ax.set_ylabel("Count")
        ax.set_title("Pixel Spacing Distribution")

    # Echo time
    with report.figure("echo_time", figsize=(10, 5)) as fig:
        ax = fig.subplots()
        sns.histplot(df["echo_time"], bins=30, ax=ax)
        ax.set_xlabel("Echo Time (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Echo Time Distribution")

    # Repetition time
    with report.figure("repetition_time", figsize=(10, 5)) as fig:
        ax = fig.subplots()
        sns.histplot(df["repetition_time"], bins=30, ax=ax)
        ax.set_xlabel("Repetition Time (ms)")
        ax.set_ylabel("Count")
        ax.set_title("Repetition Time Distribution")
