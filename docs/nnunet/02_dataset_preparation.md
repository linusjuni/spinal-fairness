# 02 — Dataset Preparation

## Required nnU-Net Directory Structure

```
$nnUNet_raw/Dataset001_CSpineSeg/
├── dataset.json
├── imagesTr/          <- train + val cases (916 from split_v3)
│   ├── cspine_000001_0000.nii.gz
│   ├── cspine_000002_0000.nii.gz
│   └── ...
├── labelsTr/          <- matching segmentation masks (no modality suffix)
│   ├── cspine_000001.nii.gz
│   ├── cspine_000002.nii.gz
│   └── ...
└── imagesTs/          <- test cases only (226 from split_v3); never used during training
    ├── cspine_001100_0000.nii.gz
    └── ...
```

> **Why no `labelsTs/`?** nnU-Net does not use test labels during training or planning.
> Test labels are kept in the original source directory and used only during fairness evaluation.
> `imagesTs/` itself is also never read by nnU-Net — it is purely a storage convention.

---

## File Naming Rules

| Component | Rule | Example |
|---|---|---|
| Case identifier | `cspine_NNNNNN` — zero-padded 6-digit patient number | `cspine_000001` |
| Modality suffix | `_0000` for all images (single-channel MRI) | `cspine_000001_0000.nii.gz` |
| Label file | Case identifier only, no modality suffix | `cspine_000001.nii.gz` |
| File ending | `.nii.gz` throughout | — |

The patient number comes from the `593973-NNNNNN` portion of the source filename
(`593973-000001_Study-MR-1_Series-22.nii.gz` -> `cspine_000001`).

> **Caveat:** Case identifiers must be identical between `imagesTr/` and `labelsTr/`.
> A mismatch causes a silent pairing failure during preprocessing.

> **Caveat:** Case identifiers must NOT end with a `_XXXX` pattern (underscore + 4 digits),
> as this is ambiguous with the channel suffix. nnU-Net extracts identifiers by stripping
> the last 5 characters (`_0000`) from image filenames. Stick to alphanumeric characters,
> underscores, and hyphens in identifiers.

---

## dataset.json

```json
{
    "channel_names": {
        "0": "MRI"
    },
    "labels": {
        "background": 0,
        "vertebral_body": 1,
        "disc": 2
    },
    "numTraining": 916,
    "file_ending": ".nii.gz"
}
```

### Required Fields

| Field | Type | Description |
|---|---|---|
| `channel_names` | `dict[str, str]` | Maps channel index (as string `"0"`, `"1"`, ...) to a channel name. The name controls normalization (see below). |
| `labels` | `dict[str, int]` | Maps label name to integer value. Must include `"background": 0`. Values must be **consecutive integers starting at 0** — gaps like `{0, 1, 3}` will fail verification. |
| `numTraining` | `int` | Count of cases in `imagesTr/` (train + val from split_v3 = 916). Must exactly match the actual case count. |
| `file_ending` | `str` | File extension used by all images and labels. |

### Optional Fields

| Field | Purpose |
|---|---|
| `overwrite_image_reader_writer` | Specify a reader/writer class (e.g., `"SimpleITKIO"`, `"NibabelIO"`). Auto-detected if omitted. |
| `regions_class_order` | List of ints for region-based training. Not needed for standard segmentation. |
| `dataset` | Dict mapping case identifiers to explicit file paths. Alternative to scanning `imagesTr/`/`labelsTr/`. |

### Channel Names and Normalization

The channel name **directly controls the normalization scheme**. The mapping is
case-insensitive:

| Channel name | Normalization | Behavior |
|---|---|---|
| `"CT"` | CTNormalization | **Global/dataset-level**: clips to [0.5th, 99.5th] percentiles of foreground intensity across all training cases, then z-score. Same normalization applied to every case. |
| `"noNorm"` | NoNormalization | No normalization; just casts to target dtype. |
| `"zscore"` | ZScoreNormalization | **Per-case**: z-score over individual image. |
| `"rescale_to_0_1"` | RescaleTo01Normalization | Per-case: rescale to [0, 1]. |
| `"rgb_to_0_1"` | RGBTo01Normalization | Assumes uint8 input. Divides by 255. |
| **Anything else** (e.g., `"MRI"`, `"T1"`, `"T2"`, `"FLAIR"`) | ZScoreNormalization | **Falls back to per-case z-score.** |

`"MRI"` is not a recognized keyword — it simply falls through to the default per-case
z-score, which is correct for our MRI data. Only the exact string `"CT"` triggers CT-specific
normalization. Do not use `"CT"` for MRI data.

---

## Split Strategy: What Goes Where

| split_v3 assignment | nnU-Net destination | Rationale |
|---|---|---|
| `train` (800 cases) | `imagesTr/` | Available for 5-fold CV |
| `val` (116 cases) | `imagesTr/` | Available for 5-fold CV |
| `test` (226 cases) | `imagesTs/` | Hard holdout — never touches training |

nnU-Net runs its own 5-fold cross-validation over `imagesTr/`. We do **not** reserve our
`val` split from nnU-Net — more data in the CV pool produces better model selection. The
`val` split in split_v3 was created for demographic balance verification, not for
withholding from training.

---

## Custom Demographic Splits (splits_final.json)

By default nnU-Net creates random 5-fold splits (using `sklearn.model_selection.KFold`
with `shuffle=True, random_state=12345`). We override this with a demographically
stratified split file so each fold maintains race x age x sex balance.

**File location** (must exist before training begins):
```
$nnUNet_preprocessed/Dataset001_CSpineSeg/splits_final.json
```

> **Caveat:** This directory is created by `nnUNetv2_plan_and_preprocess`. The
> `splits_final.json` must therefore be written **after** preprocessing but **before**
> any `nnUNetv2_train` call. If it does not exist when training starts, nnU-Net
> auto-generates random splits. The data prep script handles this automatically.

**Format:**
```json
[
  {
    "train": ["cspine_000001", "cspine_000003", ...],
    "val":   ["cspine_000002", "cspine_000009", ...]
  },
  { "train": [...], "val": [...] },
  { "train": [...], "val": [...] },
  { "train": [...], "val": [...] },
  { "train": [...], "val": [...] }
]
```

- The length of the outer list determines the number of folds (5 for standard 5-fold CV,
  but any number is valid — e.g., 3 elements for 3-fold CV).
- Values are case identifier strings matching the stems in `imagesTr/` (e.g. `"cspine_000001"`).
  **Not** full filenames — no `_0000.nii.gz` suffix.
- Generated by stratified k-fold over the 916 train+val cases using the same
  race x age x sex strata as split_v3.
- If you request a fold number beyond what the file contains (e.g., `--fold 5` with only 5
  folds defined at indices 0-4), nnU-Net prints a warning and creates a random 80:20 split
  for that fold.

---

## Running the Data Preparation Script

```bash
uv run -m src.nnunet.prepare_dataset
```

This script:
1. Reads `split_v3.tsv` to determine train/val/test assignments
2. Creates the `Dataset001_CSpineSeg/` directory structure under `$nnUNet_raw`
3. Symlinks (or copies) NIfTI files with the correct nnU-Net naming
4. Writes `dataset.json`
5. Writes the `case_id_mapping.json` (source filename <-> nnU-Net case identifier)

> **Symlinks vs copies:** nnU-Net uses standard Python file I/O and has no special symlink
> handling — symlinks work correctly as long as they resolve to valid files. Symlinking
> avoids duplicating 4.9 GB. Use copies only if the compute node cannot follow symlinks
> on work3 (check with `ls -la` on the node).

---

## Verifying the Dataset

Run integrity checks before preprocessing:

```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

The `--verify_dataset_integrity` flag checks:
- `dataset.json` exists and contains required keys (`labels`, `channel_names`, `numTraining`, `file_ending`)
- `numTraining` matches the actual number of detected cases
- Every `imagesTr` case has a matching `labelsTr` mask
- All channel files are present for every case
- Label values are consecutive integers starting at 0 (no gaps)
- No NaN values in images or segmentations
- Shape and spacing match between each image-label pair
- Affine/origin/direction consistency (warnings only, not errors)

Fix all errors reported before proceeding.
