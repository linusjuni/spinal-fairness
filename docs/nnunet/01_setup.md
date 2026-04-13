# 01 — Setup

## Installation

Install PyTorch **first** (nnunetv2 can silently replace CUDA torch with CPU-only):

```bash
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv add nnunetv2
uv add wandb
```

Known issue: nnU-Net v2.7.0 excludes `torch!=2.9.*` (3D conv regression under AMP).

## Environment Variables

Three required variables. If any is unset, nnU-Net crashes with a `RuntimeError`.

```bash
# In project .env (loaded via UV_ENV_FILE=".env" in ~/.bashrc)
nnUNet_raw=/work3/s225224/nnunet/raw
nnUNet_preprocessed=/work3/s225224/nnunet/preprocessed
nnUNet_results=/work3/s225224/nnunet/results
```

All `nnUNetv2_*` commands must be prefixed with `uv run` (entry points live inside the venv).

## Directory Layout

```
/work3/s225224/
├── data/cspineseg/                    <- source data (read-only)
│   ├── extracted/annotation/          <- 1,255 MRI volumes (.nii.gz)
│   ├── extracted/segmentation/        <- 1,255 segmentation masks (*_SEG.nii.gz)
│   └── splits/split_v3.tsv           <- train/val/test assignments
│
└── nnunet/
    ├── raw/Dataset001_CSpineSeg/      <- symlinks created by prepare_dataset
    ├── preprocessed/                  <- created by plan_and_preprocess
    └── results/                       <- created by training
```
