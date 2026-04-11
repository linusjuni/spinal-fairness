# 01 — Setup

## Installation

nnU-Net v2 is added as a dependency in `pyproject.toml`. Install with:

```bash
uv sync
```

For the DTU HPC cluster environment, ensure the virtual environment is activated before
any `nnUNetv2_*` commands.

---

## Environment Variables

Three variables are **required**. Set them permanently in `~/.bashrc`:

```bash
export nnUNet_raw="/work3/s225224/nnunet/raw"
export nnUNet_preprocessed="/work3/s225224/nnunet/preprocessed"
export nnUNet_results="/work3/s225224/nnunet/results"
```

Verify with:
```bash
echo $nnUNet_raw
echo $nnUNet_preprocessed
echo $nnUNet_results
```

> **Caveat:** These variables must be set in every shell session and in every LSF job script.
> Forgetting them causes nnU-Net to fail silently with a confusing `NoneType` path error.
> Add them to your job scripts explicitly (see [03 — Training](03_training.md)).

---

## Directory Layout on work3

```
/work3/s225224/
├── data/cspineseg/                    ← source data (read-only, do not modify)
│   ├── extracted/annotation/          ← 1,255 MRI images (.nii.gz)
│   ├── extracted/segmentation/        ← 1,255 segmentation masks (*_SEG.nii.gz)
│   └── splits/split_v3.tsv            ← our train/val/test assignments
│
└── nnunet/
    ├── raw/                           ← $nnUNet_raw
    │   └── Dataset001_CSpineSeg/      ← populated by the data prep script
    ├── preprocessed/                  ← $nnUNet_preprocessed (created by nnU-Net)
    └── results/                       ← $nnUNet_results (created by nnU-Net)
```

The `nnunet/` tree does not exist yet and will be created by the data preparation script.

---

## Dataset ID Convention

nnU-Net v2 uses `DatasetXXX_Name` where `XXX` is a zero-padded 3-digit integer.

| ID | Name |
|---|---|
| `001` | `CSpineSeg` |

Full directory name: **`Dataset001_CSpineSeg`**
