from src.data.loader import load_metadata
from src.data.volumes import load_volume_properties
from src.eda.report import EDAReport
from src.mri_visualization import plot_ortho_slices
from src.utils.settings import settings

# Load data
volumes = load_volume_properties()
metadata = load_metadata()

# Merge on series_submitter_id for demographics
df = volumes.join(metadata, on="series_submitter_id", how="left")

# Sort by total_voxels and pick 3 representative volumes
df_sorted = df.sort("total_voxels")
indices = {
    "smallest": 0,
    "median": df_sorted.height // 2,
    "largest": df_sorted.height - 1,
}

with EDAReport("mri_slices") as report:
    for label, idx in indices.items():
        row = df_sorted.row(idx, named=True)

        filename = row["filename"]
        nifti_path = settings.annotation_dir / filename

        shape = (row["width"], row["height"], row["n_slices"])
        spacing = (row["spacing_x"], row["spacing_y"], row["spacing_z"])

        output_path = report.run_dir / f"ortho_{label}.png"
        plot_ortho_slices(nifti_path, output_file=output_path)

        report.log_stat(
            label,
            {
                "filename": filename,
                "shape": list(shape),
                "spacing": [round(s, 4) for s in spacing],
                "total_voxels": row["total_voxels"],
                "age_at_imaging": row.get("age_at_imaging"),
                "sex": row.get("sex"),
                "race": row.get("race"),
            },
        )
