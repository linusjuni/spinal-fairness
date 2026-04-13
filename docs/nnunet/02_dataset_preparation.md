# 02 — Dataset Preparation

> **Status: Done.** Dataset001_CSpineSeg is built and verified.

## nnU-Net Directory Structure

```
$nnUNet_raw/Dataset001_CSpineSeg/
├── dataset.json
├── imagesTr/          <- 916 train+val cases (symlinked)
├── labelsTr/          <- matching segmentation masks
├── imagesTs/          <- 226 test cases (symlinked)
└── case_id_mapping.json
```

## File Naming

| Component | Example |
|---|---|
| Image | `cspine_000001_0000.nii.gz` |
| Label | `cspine_000001.nii.gz` |

Patient number extracted from source filename `593973-NNNNNN`. Multi-exam patients get a `_s{N}` suffix.

## dataset.json

```json
{
    "channel_names": {"0": "MRI"},
    "labels": {"background": 0, "vertebral_body": 1, "disc": 2},
    "numTraining": 916,
    "file_ending": ".nii.gz"
}
```

Channel name `"MRI"` falls through to per-case z-score normalization (correct for MRI).

## Split Strategy

| split_v3 assignment | nnU-Net destination |
|---|---|
| train (800) + val (116) | `imagesTr/` — available for 5-fold CV |
| test (226) | `imagesTs/` — never touches training |

nnU-Net's internal 5-fold CV is overridden with a demographically stratified `splits_final.json` (race x age x sex balanced). Written by `uv run -m src.nnunet.write_splits` after preprocessing.

## Rebuilding (if needed)

```bash
uv run -m src.nnunet.prepare_dataset    # build dataset dir + symlinks
uv run nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncL
uv run -m src.nnunet.write_splits       # stratified folds → splits_final.json
```
