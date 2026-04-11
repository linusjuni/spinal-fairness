# 05 — Inference & Evaluation

## Step 1: Select the Best Configuration

After all folds of all configurations have finished training (and `--npz` was used for
ensembling), run automated model selection:

```bash
nnUNetv2_find_best_configuration 1 -c 2d 3d_fullres
```

This command:
1. Accumulates cross-validation results from individual folds into a merged folder
   (`crossval_results_folds_0_1_2_3_4/`)
2. Evaluates each configuration's Dice score on cross-validation predictions
3. If ensembling is enabled (default), creates all pairwise ensemble combinations,
   averages softmax probabilities, and evaluates
4. Identifies the best single model or ensemble by foreground mean Dice
5. Runs `determine_postprocessing()` on the winning model/ensemble (tests whether
   removing all but the largest connected component improves each label)
6. Writes output files

### Key flags

| Flag | Purpose |
|---|---|
| `-c CONFIGS...` | Configurations to compare (default: `2d 3d_fullres 3d_lowres 3d_cascade_fullres`) |
| `-tr TRAINER...` | Trainer class names (default: `nnUNetTrainer`) |
| `-p PLANS...` | Plans identifiers (default: `nnUNetPlans`) |
| `-f FOLDS...` | Folds to use (default: `0 1 2 3 4`) |
| `--disable_ensembling` | Skip ensemble evaluation — only compare single models |
| `--no_overwrite` | Skip re-computing ensembles that already exist |

### `--npz` requirement

- **With ensembling (default):** `--npz` must have been used during training. The ensemble
  evaluation loads `.npz` softmax files and raises a `RuntimeError` if they are missing.
- **Without ensembling** (`--disable_ensembling`): `--npz` is not required. Only the
  segmentation files in `validation/` subfolders are needed.
- **Retroactive fix:** If you forgot `--npz`, run `nnUNetv2_train 1 2d 0 --val --npz`
  (the `--val` flag runs validation only without retraining).

### Output files

| File | Location | Contents |
|---|---|---|
| `inference_information.json` | `$nnUNet_results/Dataset001_CSpineSeg/` | All results, best model/ensemble, postprocessing info |
| `inference_instructions.txt` | `$nnUNet_results/Dataset001_CSpineSeg/` | Copy-paste commands for prediction and postprocessing |

> **Caveat:** If ensembling is enabled (default), the command evaluates combinations
> of configurations (e.g., `2d + 3d_fullres` ensemble). Ensembles almost always
> outperform single configurations. Use `--disable_ensembling` only if you want a single
> model for interpretability or compute reasons.

---

## Step 2: Predict on the Test Set

Use the command from `inference_instructions.txt`. The general form:

```bash
nnUNetv2_predict \
    -i $nnUNet_raw/Dataset001_CSpineSeg/imagesTs \
    -o $nnUNet_results/Dataset001_CSpineSeg/predictions_test \
    -d 1 \
    -c 2d \
    -f 0 1 2 3 4 \
    --save_probabilities
```

### Key flags

| Flag | Purpose |
|---|---|
| `-f 0 1 2 3 4` | Uses all 5 folds as an ensemble (averages predictions). |
| `-f all` | Uses a single model trained on all data (no CV split). |
| `--save_probabilities` | Saves `.npz` softmax outputs. Only needed if ensembling across configurations afterwards. **Large files** — omit if disk space is tight. |
| `--continue_prediction` | Skips cases where output already exists. |
| `--disable_tta` | Disables test-time augmentation (mirroring). Faster but potentially lower quality. |
| `-step_size 0.5` | Sliding window overlap (default 0.5 = 50%). Smaller = more overlap = better but slower. |
| `-chk FILENAME` | Checkpoint to use (default: `checkpoint_final.pth`). |
| `-num_parts N` / `-part_id X` | For distributed inference across multiple GPUs. |
| `-device DEVICE` | `cuda`, `cpu`, or `mps`. |

> **Caveat:** Input images in `imagesTs/` must follow the same naming convention and file
> format as `imagesTr/` (i.e., `cspine_NNNNNN_0000.nii.gz`). A format mismatch causes
> errors or empty predictions.

> **Caveat:** If `find_best_configuration` selected `3d_cascade_fullres`, you must first
> predict with `3d_lowres`, then pass its output via `-prev_stage_predictions FOLDER`.
> This is unlikely for our data but check `inference_instructions.txt`.

### Multi-GPU inference

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict [...] -num_parts 2 -part_id 0 &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict [...] -num_parts 2 -part_id 1
```

---

## Step 3: Apply Postprocessing

nnU-Net learns postprocessing rules during `find_best_configuration` (e.g., keep only the
largest connected component per label). Apply these to the raw predictions:

```bash
nnUNetv2_apply_postprocessing \
    -i $nnUNet_results/Dataset001_CSpineSeg/predictions_test \
    -o $nnUNet_results/Dataset001_CSpineSeg/predictions_test_pp \
    -pp_pkl_file $nnUNet_results/Dataset001_CSpineSeg/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl \
    -plans_json $nnUNet_preprocessed/Dataset001_CSpineSeg/nnUNetPlans.json \
    -dataset_json $nnUNet_raw/Dataset001_CSpineSeg/dataset.json
```

> **Note:** The flags use single dashes (`-pp_pkl_file`, `-plans_json`, `-dataset_json`),
> not double dashes. The exact path to `postprocessing.pkl` is printed by
> `find_best_configuration` and written to `inference_instructions.txt`. Use what is
> printed there, not a hardcoded path.

> **Note:** If `-plans_json` and `-dataset_json` are omitted, the code looks for
> `plans.json` and `dataset.json` inside the input folder (`-i`). For single-model
> predictions, nnU-Net copies these files into the output folder automatically.

---

## Step 4: Compute Segmentation Metrics

nnU-Net v2 includes evaluation scripts that compute **Dice and IoU** per case and per label:

```bash
nnUNetv2_evaluate_folder \
    $SOURCE_SEGMENTATION_DIR \
    $nnUNet_results/Dataset001_CSpineSeg/predictions_test_pp \
    -djfile $nnUNet_raw/Dataset001_CSpineSeg/dataset.json \
    -pfile $nnUNet_preprocessed/Dataset001_CSpineSeg/nnUNetPlans.json
```

> **Note:** The first two arguments are **positional** (ground-truth folder, prediction
> folder), not named flags like `-ref`/`-pred`.

Where `$SOURCE_SEGMENTATION_DIR` contains the ground-truth test masks (from
`/work3/s225224/data/cspineseg/extracted/segmentation/`), renamed to match prediction
filenames.

There is also a simpler variant that takes label IDs directly:
```bash
nnUNetv2_evaluate_simple \
    $SOURCE_SEGMENTATION_DIR \
    $nnUNet_results/Dataset001_CSpineSeg/predictions_test_pp \
    -l 1 2
```

### Metrics computed

Per label, per case:
- **Dice coefficient**: `2*TP / (2*TP + FP + FN)`
- **IoU (Jaccard)**: `TP / (TP + FP + FN)`
- **FP, TP, FN, TN** (raw counts)

> **Important: Hausdorff Distance 95 (HD95) is NOT computed by default** in nnU-Net v2's
> evaluation. HD95 was part of v1 but was removed from the default metrics in v2. If you
> need HD95 for the fairness analysis, compute it separately using a library like
> `surface-distance` or `medpy`:
>
> ```bash
> uv add surface-distance
> ```
>
> See Step 5 for how to integrate HD95 into the fairness analysis.

### Output format

JSON file (`summary.json` in the prediction folder) with structure:

```json
{
    "metric_per_case": [
        {
            "reference_file": "...",
            "prediction_file": "...",
            "metrics": {
                "1": {"Dice": 0.95, "IoU": 0.91, "FP": 123, "TP": 45678, ...},
                "2": {"Dice": 0.88, "IoU": 0.79, ...}
            }
        }
    ],
    "mean": {
        "1": {"Dice": 0.94, ...},
        "2": {"Dice": 0.87, ...}
    },
    "foreground_mean": {
        "Dice": 0.91,
        "IoU": 0.85
    }
}
```

Additional flags:
- `-o PATH` — custom output path for `summary.json`
- `--chill` — tolerate missing predictions (useful for partial evaluations)

---

## Step 5: Fairness Analysis

Join `summary.json` metrics with `split_v3.tsv` demographics:

```python
# Pseudocode — see src/fairness/ (to be implemented)
metrics = load_summary_json(...)          # case_id -> {dice_vb, dice_disc, iou_vb, ...}
split   = load_split_v3(...)             # case_id -> {sex, race_bin, age_bin, ...}
df      = metrics.join(split, on="case_id")

# Compare across subgroups
for group_col in ["sex_bin", "race_bin", "age_bin"]:
    for metric in ["dice_vb", "dice_disc"]:
        run_mann_whitney(df, group_col, metric)
        plot_boxplot(df, group_col, metric)
```

### HD95 computation (separate from nnU-Net)

Since nnU-Net v2 does not compute HD95 by default, add it in the fairness analysis:

```python
from surface_distance import compute_surface_distances, compute_robust_hausdorff

def compute_hd95(pred_mask, gt_mask, spacing_mm):
    distances = compute_surface_distances(gt_mask, pred_mask, spacing_mm)
    return compute_robust_hausdorff(distances, 95)
```

### Key analysis priorities (from EDA findings)

| Comparison | Primary metric | Known confounder |
|---|---|---|
| Female vs Male | Dice (both labels) | Males have ~25% larger structures -> volume-adjust before attributing gaps |
| White vs Black | Dice disc (cleaner target) | Age skew -> stratify or include as covariate |
| Age bins | Dice both labels | -- |
| Gold vs Silver | Dice both labels | Pending author data |

### Published baseline

The CSpineSeg dataset authors (Zhai et al., *Scientific Data* 2025) reported ensemble
Dice of **0.929** (vertebral body), **0.904** (disc), **0.916** (macro-average). Our
model's overall performance should be in this ballpark. Sub-group performance gaps on top
of this baseline are the fairness signal we are looking for.

---

## Caveats

- **Test set is sacred.** Never run `nnUNetv2_predict` on the test set until training and
  model selection are fully complete. Any peeking — even just running inference to check
  it works — invalidates the holdout.
- **HD95 is not in nnU-Net v2 by default.** Must be computed separately. HD95 is sensitive
  to single outlier voxels. Report alongside Dice; treat HD95 anomalies as signals to
  inspect individual cases.
- **Wide-FOV outliers.** 22 exams with >15 components (mostly White females on Siemens
  3T) may inflate error metrics. Flag these in the fairness analysis — they are not
  necessarily a fairness signal.
- **Volume adjustment.** Raw Dice comparisons between male and female cases are confounded
  by ~25% anatomical size difference. Consider normalising metrics by vertebral body
  volume (mm3) before reporting group gaps.
- **Gold/Silver distinction pending.** Until the Duke authors provide expert-annotated
  case IDs, all labels are treated as equivalent. The fairness analysis must be re-run
  once the distinction is available — this is a primary research question of the project.
