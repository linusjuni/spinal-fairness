# 05 — Model Selection & Test Evaluation

> **Status (2026-05-01):** Predict jobs submitted (28332649, 28332650) to gpul40s — pending in queue. labelsTs directories created.

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

Two GPU jobs in parallel (~1–2 hours each on L40s for 228 test cases). Use `jobs/predict.sh`:

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

## Step 4 — Create labelsTs directories

Three reference directories are needed — same predictions are evaluated against each:

| Directory | Cases | Purpose |
|---|---|---|
| `labelsTs/` | 226 (all test) | Compare against published baseline (Zhou et al.) |
| `labelsTs_gold/` | 76 (gold test) | True quality — expert labels only |
| `labelsTs_silver/` | 138 (silver test) | Biased ruler — auto-generated labels only |

`labelsTs_gold/` and `labelsTs_silver/` use the case IDs from `split_v3_gold.tsv` and `split_v3_silver.tsv` respectively to select which symlinks to create.

```bash
uv run --env-file .env python - <<'EOF'
import json
import polars as pl
from src.utils.settings import settings

mapping = json.loads((settings.nnUNet_raw / "Dataset001_CSpineSeg/case_id_mapping.json").read_text())

gold_ids  = set(pl.read_csv(settings.splits_dir / "split_v3_gold.tsv",   separator="\t")
                  .filter(pl.col("split") == "test")["series_submitter_id"].to_list())
silver_ids = set(pl.read_csv(settings.splits_dir / "split_v3_silver.tsv", separator="\t")
                   .filter(pl.col("split") == "test")["series_submitter_id"].to_list())

dirs = {
    "labelsTs":        lambda e: True,
    "labelsTs_gold":   lambda e: e["series_submitter_id"] in gold_ids,
    "labelsTs_silver": lambda e: e["series_submitter_id"] in silver_ids,
}

for dirname, pred in dirs.items():
    out = settings.nnUNet_raw / f"Dataset001_CSpineSeg/{dirname}"
    out.mkdir(exist_ok=True)
    n = 0
    for entry in mapping:
        if entry["split"] != "test" or not pred(entry):
            continue
        src = settings.segmentation_dir / entry["source_filename"].replace(".nii.gz", "_SEG.nii.gz")
        dst = out / f"{entry['case_id']}.nii.gz"
        if not dst.exists() and src.exists():
            dst.symlink_to(src.resolve())
            n += 1
    print(f"{dirname}: linked {n} labels")
EOF
```

---

## Step 5 — Evaluate on Test Set

CPU only. Run three evaluations against the same `predictions_test_pp/`:

```bash
# Mixed — comparison with published baseline (Zhou et al.: VB 0.929, disc 0.904, macro 0.916)
uv run --env-file .env nnUNetv2_evaluate_folder \
    ${nnUNet_raw}/Dataset001_CSpineSeg/labelsTs \
    ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp \
    -djfile ${nnUNet_raw}/Dataset001_CSpineSeg/dataset.json \
    -pfile  ${nnUNet_preprocessed}/Dataset001_CSpineSeg/nnUNetResEncUNetLPlans.json

# Gold — true quality against expert labels
uv run --env-file .env nnUNetv2_evaluate_folder \
    ${nnUNet_raw}/Dataset001_CSpineSeg/labelsTs_gold \
    ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp \
    -djfile ${nnUNet_raw}/Dataset001_CSpineSeg/dataset.json \
    -pfile  ${nnUNet_preprocessed}/Dataset001_CSpineSeg/nnUNetResEncUNetLPlans.json \
    -o      ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp/summary_gold.json

# Silver — biased ruler effect
uv run --env-file .env nnUNetv2_evaluate_folder \
    ${nnUNet_raw}/Dataset001_CSpineSeg/labelsTs_silver \
    ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp \
    -djfile ${nnUNet_raw}/Dataset001_CSpineSeg/dataset.json \
    -pfile  ${nnUNet_preprocessed}/Dataset001_CSpineSeg/nnUNetResEncUNetLPlans.json \
    -o      ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp/summary_silver.json
```

If gold Dice > silver Dice on the same predictions, silver labels are an optimistic (biased) ruler.

### HD95

`nnUNetv2_evaluate_folder` only computes Dice and IoU. HD95 must be computed separately using the `surface-distance` package. Run for all three reference sets to match the metrics used in Aditya et al. (HD95 and Dice can show different bias patterns).

```bash
uv run --env-file .env python -m src.fairness.compute_hd95 \
    --predictions ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp \
    --references  ${nnUNet_raw}/Dataset001_CSpineSeg/labelsTs \
    --output      ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp/hd95.csv
```

(Script `src/fairness/compute_hd95.py` not yet written — to be implemented before fairness analysis.)

---

## Step 6 — Fairness Analysis

Implementation in `src/fairness/` (not yet built). Joins per-case Dice and HD95 metrics with demographics from `split_v3.tsv` via `case_id_mapping.json`. Runs separately for gold and silver reference labels to test whether fairness conclusions change with the ruler. See `docs/nnunet/04_inference.md` for confounder notes and `docs/statistical-testing/` for planned statistical tests.
