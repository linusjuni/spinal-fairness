# 05 — Model Selection & Test Evaluation

> **Status (2026-04-30):** Training complete. 3 validation runs pending on gpul40s (jobs 28331042–28331044). Blocked on those before model selection can proceed.

## Current State

### Training (done)

All 10 jobs finished — every fold has `checkpoint_final.pth`:

```
$nnUNet_results/Dataset001_CSpineSeg/
├── nnUNetTrainerWandB__nnUNetResEncUNetLPlans__2d/
│   ├── fold_0/  checkpoint_final.pth ✅  validation/summary.json ✅
│   ├── fold_1/  checkpoint_final.pth ✅  validation/summary.json ✅
│   ├── fold_2/  checkpoint_final.pth ✅  validation/summary.json ✅
│   ├── fold_3/  checkpoint_final.pth ✅  validation/summary.json ✅
│   └── fold_4/  checkpoint_final.pth ✅  validation/summary.json ⏳ (job 28331042)
└── nnUNetTrainerWandB__nnUNetResEncUNetLPlans__3d_fullres/
    ├── fold_0/  checkpoint_final.pth ✅  validation/summary.json ✅
    ├── fold_1/  checkpoint_final.pth ✅  validation/summary.json ⏳ (job 28331043)
    ├── fold_2/  checkpoint_final.pth ✅  validation/summary.json ✅
    ├── fold_3/  checkpoint_final.pth ✅  validation/summary.json ⏳ (job 28331044)
    └── fold_4/  checkpoint_final.pth ✅  validation/summary.json ✅
```

3 folds are missing `validation/summary.json` because their jobs crashed at the end with `OSError: No space left on device` during WandB cleanup (training itself completed fine). Re-run jobs submitted 2026-04-30 via `jobs/validate.sh` to `gpul40s`.

### CV Validation Dice (not test — not comparable to published baseline yet)

| Config | Fold | VB Dice | Disc Dice | Macro |
|---|---|---|---|---|
| 2d | 0 | 0.9614 | 0.9352 | 0.9483 |
| 2d | 1 | 0.9517 | 0.9220 | 0.9369 |
| 2d | 2 | 0.9539 | 0.9216 | 0.9377 |
| 2d | 3 | 0.9571 | 0.9301 | 0.9436 |
| 2d | 4 | — | — | — |
| 3d_fullres | 0 | 0.9586 | 0.9334 | 0.9460 |
| 3d_fullres | 1 | — | — | — |
| 3d_fullres | 2 | 0.9509 | 0.9182 | 0.9346 |
| 3d_fullres | 3 | — | — | — |
| 3d_fullres | 4 | 0.9491 | 0.9213 | 0.9352 |

2d is consistently slightly ahead of 3d_fullres on available folds. `find_best_configuration` will confirm whether to use 2d alone or an ensemble.

---

## Step 1 — Find Best Configuration

Once all 10 `summary.json` files exist, run on a **login node** (CPU only, ~5 min):

```bash
cd /zhome/77/2/187952/projects/spinal-fairness
source .env
uv run nnUNetv2_find_best_configuration 1 -c 2d 3d_fullres
```

This compares mean CV Dice for each config (2d, 3d_fullres) and their ensemble, determines per-config postprocessing rules (e.g. keep-largest-component), and writes the result to:

```
$nnUNet_results/Dataset001_CSpineSeg/inference_instructions.txt
```

Read that file — it contains exact copy-paste commands for the predict and postprocessing steps.

---

## Step 2 — Predict on Test Set

Submit as a bsub job to `gpul40s` (GPU required, ~1–2 hours for 417 cases). Use the command from `inference_instructions.txt`. General form:

```bash
uv run nnUNetv2_predict \
    -i $nnUNet_raw/Dataset001_CSpineSeg/imagesTs \
    -o $nnUNet_results/Dataset001_CSpineSeg/predictions_test \
    -d 1 \
    -c <CONFIG> \
    -f 0 1 2 3 4 \
    -p nnUNetResEncUNetLPlans \
    -tr nnUNetTrainerWandB
```

If the best config is an ensemble, `inference_instructions.txt` will give two `nnUNetv2_predict` calls (one per config) plus an ensemble step.

---

## Step 3 — Apply Postprocessing

CPU only, fast. Command from `inference_instructions.txt`:

```bash
uv run nnUNetv2_apply_postprocessing \
    -i $nnUNet_results/Dataset001_CSpineSeg/predictions_test \
    -o $nnUNet_results/Dataset001_CSpineSeg/predictions_test_pp \
    -pp_pkl_file <PATH_FROM_INSTRUCTIONS> \
    -plans_json $nnUNet_preprocessed/Dataset001_CSpineSeg/nnUNetResEncUNetLPlans.json \
    -dataset_json $nnUNet_raw/Dataset001_CSpineSeg/dataset.json
```

---

## Step 4 — Create labelsTs

Test labels were not symlinked into the nnUNet dataset directory (they stay in the source segmentation directory). Create them once before evaluation:

```bash
source .env
uv run python - <<'EOF'
import json
from pathlib import Path
from src.utils.settings import settings

mapping = json.loads((settings.nnUNet_raw / "Dataset001_CSpineSeg/case_id_mapping.json").read_text())
labels_ts = settings.nnUNet_raw / "Dataset001_CSpineSeg/labelsTs"
labels_ts.mkdir(exist_ok=True)

n = 0
for entry in mapping:
    if entry["split"] != "test":
        continue
    src = settings.segmentation_dir / entry["source_filename"].replace(".nii.gz", "_SEG.nii.gz")
    dst = labels_ts / f"{entry['case_id']}.nii.gz"
    if not dst.exists() and src.exists():
        dst.symlink_to(src.resolve())
        n += 1
print(f"Linked {n} test labels")
EOF
```

---

## Step 5 — Evaluate on Test Set

CPU only. Produces `summary.json` with per-case Dice and IoU for labels 1 (vertebral body) and 2 (disc):

```bash
uv run nnUNetv2_evaluate_folder \
    $nnUNet_raw/Dataset001_CSpineSeg/labelsTs \
    $nnUNet_results/Dataset001_CSpineSeg/predictions_test_pp \
    -djfile $nnUNet_raw/Dataset001_CSpineSeg/dataset.json \
    -pfile $nnUNet_preprocessed/Dataset001_CSpineSeg/nnUNetResEncUNetLPlans.json
```

Output: `predictions_test_pp/summary.json` — per-case Dice for all test cases. This is the first number directly comparable to the published baseline (Zhou et al.: VB 0.929, disc 0.904, macro 0.916).

HD95 is not computed by `nnUNetv2_evaluate_folder` — compute separately with `surface-distance` for the fairness analysis.

---

## Step 6 — Fairness Analysis

Implementation in `src/fairness/` (not yet built). Joins per-case metrics with demographics from `split_v3.tsv` via `case_id_mapping.json`. See `docs/nnunet/04_inference.md` for confounder notes and `docs/statistical-testing/` for planned statistical tests.
