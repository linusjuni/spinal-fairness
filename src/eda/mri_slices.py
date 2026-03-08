from src.data.loader import load_metadata
from src.data.schemas import Col, VolumeCol
from src.data.volumes import load_volume_properties
from src.eda.report import EDAReport
from src.mri_visualization import plot_mri
from src.utils.settings import settings

# Load data
volumes = load_volume_properties()
metadata = load_metadata()

# Merge on series_submitter_id for demographics
df = volumes.join(metadata, on=Col.SERIES_SUBMITTER_ID, how="left")

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

        filename = row[Col.FILENAME]
        nifti_path = settings.annotation_dir / filename

        shape = (row[VolumeCol.WIDTH], row[VolumeCol.HEIGHT], row[VolumeCol.N_SLICES])
        spacing = (row[VolumeCol.SPACING_X], row[VolumeCol.SPACING_Y], row[VolumeCol.SPACING_Z])

        output_path = report.run_dir / f"sagittal_{label}.png"
        plot_mri(nifti_path, output_file=output_path)

        report.log_stat(
            label,
            {
                Col.FILENAME: filename,
                "shape": list(shape),
                "spacing": [round(s, 4) for s in spacing],
                VolumeCol.TOTAL_VOXELS: row[VolumeCol.TOTAL_VOXELS],
                Col.AGE: row.get(Col.AGE),
                Col.SEX: row.get(Col.SEX),
                Col.RACE: row.get(Col.RACE),
            },
        )
