# 05 — Model Selection & Test Evaluation

> **Status (2026-05-01):** Complete through Step 5. Eval CSVs written to `predictions_test_pp/`.

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

### Test Set Dice (done 2026-05-01)

Ensemble predictions after postprocessing (`predictions_test_pp/`), evaluated against three reference sets.

| Reference set | N | VB Dice | Disc Dice | Macro |
|---|---|---|---|---|
| all (mixed labels) | 228 | 0.9594 | 0.9353 | 0.9473 |
| gold (expert labels) | 76 | 0.9216 | 0.8721 | 0.8969 |
| silver (auto labels) | 138 | 0.9810 | 0.9723 | 0.9766 |

**Comparison with Zhou et al. baseline** (trained on 391 gold, tested on 100 gold, standard nnU-Net ensemble):

| | VB | Disc | Macro |
|---|---|---|---|
| Zhou et al. ensemble | 0.929±0.048 | 0.904±0.045 | 0.916±0.012 |
| **Ours (gold ref)** | **0.922** | **0.872** | **0.897** |
| Ours (all ref) | 0.959 | 0.935 | 0.947 |

The apples-to-apples comparison is gold vs gold: we are ~2 points below their ensemble despite using a larger architecture (ResEncUNetL) and more total training data (798 vs 391 cases). The "all" number (0.947) exceeds theirs but is inflated by silver labels, which the model scores very high against.

The gold/silver gap (0.897 vs 0.977 macro) is the first observable signal of the biased ruler effect: the same predictions score ~8 points higher when measured against auto-generated labels than expert labels. The disc label shows the largest spread (0.872 gold vs 0.972 silver), consistent with auto-labelers struggling most at disc boundaries.

Macro Dice distribution on all 228 cases: p5=0.853, p25=0.920, p50=0.975, p75=0.982, p95=0.987, min=0.483.

---

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

## Step 2 — Predict on Test Set (done 2026-05-01)

Two GPU jobs in parallel on `gpul40s`. Jobs 28332649 (2d, ~1h2m) and 28332650 (3d_fullres, ~48min) both completed successfully. Each predicted all 417 cases (228 test + 189 that were in the source folder but not the test split — nnUNet predicts everything in `imagesTs/`).

```bash
bash jobs/submit_predict.sh 1   # or: sed 's/TPLCONFIG/2d/g' jobs/predict.sh | bsub
```

Each job writes predictions + `.npz` softmax files to:
- `$nnUNet_results/Dataset001_CSpineSeg/predictions_test_2d/`
- `$nnUNet_results/Dataset001_CSpineSeg/predictions_test_3d_fullres/`

---

## Step 3 — Ensemble + Postprocessing (done 2026-05-01)

CPU only, run on login node. Requires `source .env` first so `$nnUNet_results` is available to bash before `uv run` expands it.

```bash
source .env
uv run nnUNetv2_ensemble -i ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_2d ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_3d_fullres -o ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_ensemble -np 8
uv run nnUNetv2_apply_postprocessing -i ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_ensemble -o ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp -pp_pkl_file ${nnUNet_results}/Dataset001_CSpineSeg/ensembles/ensemble___nnUNetTrainerWandB__nnUNetResEncUNetLPlans__2d___nnUNetTrainerWandB__nnUNetResEncUNetLPlans__3d_fullres___0_1_2_3_4/postprocessing.pkl -np 8 -plans_json ${nnUNet_results}/Dataset001_CSpineSeg/ensembles/ensemble___nnUNetTrainerWandB__nnUNetResEncUNetLPlans__2d___nnUNetTrainerWandB__nnUNetResEncUNetLPlans__3d_fullres___0_1_2_3_4/plans.json
```

Output: `$nnUNet_results/Dataset001_CSpineSeg/predictions_test_pp/` (final `.nii.gz` segmentations).

---

## Step 4 — Create labelsTs directories

Three reference directories are needed — same predictions are evaluated against each:

| Directory | Cases | Purpose |
|---|---|---|
| `labelsTs/` | 226 (all test) | Compare against published baseline (Zhou et al.) |
| `labelsTs_gold/` | 76 (gold test) | Expert labels only — reference for cross-model comparisons |
| `labelsTs_silver/` | 138 (silver test) | Auto-generated labels only — descriptive comparison |

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

## Step 5 — Evaluate on Test Set (done 2026-05-01)

CPU only. Uses `src.fairness.evaluate` (not `nnUNetv2_evaluate_folder`) because it outputs per-case CSVs that feed directly into `src.fairness.analyze`, and computes both Dice and HD95 in one pass. HD95 is deferred for now.

```bash
source .env
uv run -m src.fairness.evaluate --predictions ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp --references ${nnUNet_raw}/Dataset001_CSpineSeg/labelsTs --mapping ${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json --output ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp/eval_all.csv
uv run -m src.fairness.evaluate --predictions ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp --references ${nnUNet_raw}/Dataset001_CSpineSeg/labelsTs_gold --mapping ${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json --output ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp/eval_gold.csv
uv run -m src.fairness.evaluate --predictions ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp --references ${nnUNet_raw}/Dataset001_CSpineSeg/labelsTs_silver --mapping ${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json --output ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp/eval_silver.csv
```

Comparing gold vs silver Dice on the same predictions gives a descriptive sense of label
quality differences. The clean biased ruler experiment evaluates Dataset001 against both
gold labels and Dataset002's predictions (generated silver) on the same 76 gold test
images — see `06_gold_silver_training.md`.

### Results

| Reference set | Cases evaluated | Purpose |
|---|---|---|
| `labelsTs` (all) | 228 | Comparison with Zhou et al. baseline |
| `labelsTs_gold` | 76 | Expert-label quality |
| `labelsTs_silver` | 138 | Auto-label quality |

### Expected warnings

The gold and silver runs emit `[WARNING] Missing reference case_id=...` for most of the 228 cases — this is expected. `case_id_mapping.json` lists all 228 v3 test cases, but `labelsTs_gold` and `labelsTs_silver` only contain the subset belonging to each quality tier. The evaluator skips missing references and counts only the ones present.

Case accounting:

```
v3 test total:         228
v3_gold test:           76   (missing from gold eval = 152)
v3_silver test:        138   (missing from silver eval = 90)
in neither subset:      14   (sex-rebalancing dropped these from both gold and silver splits;
                              they still have labels in labelsTs and appear in eval_all.csv)

76 + 138 + 14 = 228  ✓
```

The 14 cases present in `labelsTs` but absent from both `labelsTs_gold` and `labelsTs_silver`:
`cspine_000057`, `cspine_000134`, `cspine_000314`, `cspine_000364`, `cspine_000384`,
`cspine_000435`, `cspine_000459`, `cspine_000495`, `cspine_000524`, `cspine_000659`,
`cspine_000886`, `cspine_000980`, `cspine_001033`, `cspine_001088`.

---

## Step 6 — Fairness Analysis

Implementation in `src/fairness/` (not yet built). Joins per-case Dice and HD95 metrics
with demographics from `split_v3.tsv` via `case_id_mapping.json`. This is the global
fairness audit of the mixed-trained model. The biased ruler and bias amplification
experiments use Dataset002/003 — see `06_gold_silver_training.md`. See
`docs/nnunet/04_inference.md` for confounder notes and `docs/statistical-testing/` for
planned statistical tests.
