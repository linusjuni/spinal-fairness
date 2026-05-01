# 05 — Model Selection & Test Evaluation

> **Status (2026-05-01):** All 10 validation folds complete. `find_best_configuration` done — ensemble (2d + 3d_fullres) selected. Predict jobs not yet submitted.

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
│   └── fold_4/  checkpoint_final.pth ✅  validation/summary.json ✅
└── nnUNetTrainerWandB__nnUNetResEncUNetLPlans__3d_fullres/
    ├── fold_0/  checkpoint_final.pth ✅  validation/summary.json ✅
    ├── fold_1/  checkpoint_final.pth ✅  validation/summary.json ✅
    ├── fold_2/  checkpoint_final.pth ✅  validation/summary.json ✅
    ├── fold_3/  checkpoint_final.pth ✅  validation/summary.json ✅
    └── fold_4/  checkpoint_final.pth ✅  validation/summary.json ✅
```

### CV Validation Dice (not test — not comparable to published baseline yet)

| Config | Fold | VB Dice | Disc Dice | Macro |
|---|---|---|---|---|
| 2d | 0 | 0.9614 | 0.9352 | 0.9483 |
| 2d | 1 | 0.9517 | 0.9220 | 0.9369 |
| 2d | 2 | 0.9539 | 0.9216 | 0.9377 |
| 2d | 3 | 0.9571 | 0.9301 | 0.9436 |
| 2d | 4 | 0.9512 | 0.9230 | 0.9371 |
| **2d mean** | | **0.9551** | **0.9264** | **0.9407** |
| 3d_fullres | 0 | 0.9586 | 0.9334 | 0.9460 |
| 3d_fullres | 1 | 0.9485 | 0.9220 | 0.9352 |
| 3d_fullres | 2 | 0.9509 | 0.9182 | 0.9346 |
| 3d_fullres | 3 | 0.9536 | 0.9282 | 0.9409 |
| 3d_fullres | 4 | 0.9491 | 0.9213 | 0.9352 |
| **3d_fullres mean** | | **0.9521** | **0.9246** | **0.9384** |

2d leads 3d_fullres by ~0.002 macro Dice across all folds.

---

## Step 1 — Find Best Configuration (done 2026-05-01)

```bash
uv run --env-file .env nnUNetv2_find_best_configuration 1 -c 2d 3d_fullres -p nnUNetResEncUNetLPlans -tr nnUNetTrainerWandB
```

**Result: ensemble (2d + 3d_fullres) wins.**

| Config | CV Macro Dice |
|---|---|
| 2d | 0.9407 |
| 3d_fullres | 0.9384 |
| **ensemble** | **0.9430** |

**Postprocessing determined:**
- Keep-largest-foreground globally: ✅ applied (0.94298 → 0.94309, tiny gain)
- Keep-largest per label 1 (vertebral body): ✗ not applied (0.957 → 0.491 — destroys all but one segment)
- Keep-largest per label 2 (disc): ✗ not applied (0.929 → 0.252 — same reason)

Postprocessing pkl:
```
$nnUNet_results/Dataset001_CSpineSeg/ensembles/ensemble___nnUNetTrainerWandB__nnUNetResEncUNetLPlans__2d___nnUNetTrainerWandB__nnUNetResEncUNetLPlans__3d_fullres___0_1_2_3_4/postprocessing.pkl
```

---

## Step 2 — Predict on Test Set

Two GPU jobs in parallel (~1–2 hours each on L40s for 226 test cases). Use `jobs/predict.sh`:

```bash
sed 's/TPLCONFIG/2d/g'         jobs/predict.sh | bsub
sed 's/TPLCONFIG/3d_fullres/g' jobs/predict.sh | bsub
```

Each job writes predictions + `.npz` softmax files to:
- `$nnUNet_results/Dataset001_CSpineSeg/predictions_test_2d/`
- `$nnUNet_results/Dataset001_CSpineSeg/predictions_test_3d_fullres/`

---

## Step 3 — Ensemble + Postprocessing

CPU only, run on login node after both predict jobs finish:

```bash
# Ensemble (average softmax probabilities)
uv run --env-file .env nnUNetv2_ensemble \
    -i ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_2d \
       ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_3d_fullres \
    -o ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_ensemble \
    -np 8

# Apply postprocessing (keep-largest-foreground)
uv run --env-file .env nnUNetv2_apply_postprocessing \
    -i  ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_ensemble \
    -o  ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp \
    -pp_pkl_file ${nnUNet_results}/Dataset001_CSpineSeg/ensembles/ensemble___nnUNetTrainerWandB__nnUNetResEncUNetLPlans__2d___nnUNetTrainerWandB__nnUNetResEncUNetLPlans__3d_fullres___0_1_2_3_4/postprocessing.pkl \
    -np 8 \
    -plans_json ${nnUNet_results}/Dataset001_CSpineSeg/ensembles/ensemble___nnUNetTrainerWandB__nnUNetResEncUNetLPlans__2d___nnUNetTrainerWandB__nnUNetResEncUNetLPlans__3d_fullres___0_1_2_3_4/plans.json
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
