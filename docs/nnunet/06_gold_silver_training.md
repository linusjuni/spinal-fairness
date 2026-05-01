# 06 — Gold / Silver Label Experiment

> **Status (2026-04-30):** Splits ready. Infrastructure build in progress. Training not yet submitted.

## Context

The **biased ruler** experiment (Act 3) is already handled in `05_model_selection.md`:
the Dataset001 mixed-trained model's predictions are evaluated against three reference
sets (all labels, gold-only, silver-only) to show that the choice of ruler changes the
observed fairness gap. No additional training is needed for that.

This document covers **bias amplification** (Act 5): does training on silver labels
*widen* the fairness gap? For this we train two additional models — one on gold labels
only, one on silver labels only — and compare all three models' fairness gaps when
evaluated against the same ground truth (gold test labels).

## Overview

| Dataset | Train cases | Labels | Split file | Status |
|---|---|---|---|---|
| `Dataset001_CSpineSeg` | 800 (318 gold + 480 silver) | Mixed | `split_v3` | Trained |
| `Dataset002_CSpineSeg_Gold` | 288 | Expert (gold) | `split_v3_gold` | To build |
| `Dataset003_CSpineSeg_Silver` | 450 | Auto-generated (silver) | `split_v3_silver` | To build |

All three use sex-balanced cohorts (50/50 M/F in every split).

## Evaluation design

All three models are evaluated on the **gold test set** (76 cases from `split_v3_gold`)
against expert labels. This is the apples-to-apples comparison — same test cases, same
ground truth, different training labels.

| Comparison | What it shows |
|---|---|
| Dataset001 (mixed) vs Dataset002 (gold) on gold test | Does including silver labels in training hurt fairness? |
| Dataset001 (mixed) vs Dataset003 (silver) on gold test | Does training on silver-only amplify the fairness gap? |
| Dataset002 (gold) vs Dataset003 (silver) on gold test | Clean isolation: gold-trained vs silver-trained, same eval |

If the silver-trained model shows wider demographic gaps than the gold-trained model
(cf. Parikh et al. MAMA-MIA: 66% DIR widening), silver labels are not just a biased
ruler — they actively amplify bias through the training loop.

---

## Step 1 — Extend `prepare_dataset.py`

Currently `src/nnunet/__init__.py` hardcodes:
```python
DATASET_ID = 1
DATASET_NAME = "Dataset001_CSpineSeg"
SPLIT_VERSION = "split_v3"
```

These need to become parameters. Modify `prepare_dataset.py` to accept `--dataset-id` and `--annotation-quality` CLI arguments, then run:

```bash
# Gold dataset
uv run -m src.nnunet.prepare_dataset --dataset-id 2 --annotation-quality gold

# Silver dataset
uv run -m src.nnunet.prepare_dataset --dataset-id 3 --annotation-quality silver
```

This creates:
```
$nnUNet_raw/
├── Dataset002_CSpineSeg_Gold/
│   ├── imagesTr/   (288 train + 44 val cases, symlinked)
│   ├── labelsTr/   (expert labels only)
│   ├── imagesTs/   (76 test cases)
│   └── dataset.json
└── Dataset003_CSpineSeg_Silver/
    ├── imagesTr/   (450 train + 66 val cases, symlinked)
    ├── labelsTr/   (auto-generated labels only)
    ├── imagesTs/   (138 test cases)
    └── dataset.json
```

---

## Step 2 — Plan and Preprocess

CPU only (~10–20 min each). Run on login node:

```bash
uv run nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
uv run nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity
```

---

## Step 3 — Write CV Splits

Same stratified 5-fold logic as `write_splits.py` but using the new split files:

```bash
uv run -m src.nnunet.write_splits --dataset-id 2 --split-version split_v3_gold
uv run -m src.nnunet.write_splits --dataset-id 3 --split-version split_v3_silver
```

Writes `splits_final.json` to each preprocessed dataset directory.

---

## Step 4 — Submit Training Jobs

Same `jobs/train.sh` template, just different dataset IDs. Train both 2d and 3d_fullres
for consistency with Dataset001 — decide on best config per dataset after training via
`find_best_configuration`:

```bash
# Dataset002 — Gold (10 jobs, smaller dataset so faster)
for FOLD in 0 1 2 3 4; do
  sed "s/TPLCONFIG/2d/g;        s/TPLFOLD/${FOLD}/g" jobs/train.sh \
    | sed 's/DATASET_ID=1/DATASET_ID=2/' | bsub
  sed "s/TPLCONFIG/3d_fullres/g; s/TPLFOLD/${FOLD}/g" jobs/train.sh \
    | sed 's/DATASET_ID=1/DATASET_ID=2/' | bsub
done

# Dataset003 — Silver
for FOLD in 0 1 2 3 4; do
  sed "s/TPLCONFIG/2d/g;        s/TPLFOLD/${FOLD}/g" jobs/train.sh \
    | sed 's/DATASET_ID=1/DATASET_ID=3/' | bsub
  sed "s/TPLCONFIG/3d_fullres/g; s/TPLFOLD/${FOLD}/g" jobs/train.sh \
    | sed 's/DATASET_ID=1/DATASET_ID=3/' | bsub
done
```

Or add a `--dataset-id` parameter to `train.sh` and create a `submit_gold_silver.sh`.

---

## Step 5 — Evaluate on Gold Test Set

After training and `find_best_configuration` for each dataset, predict on the gold test
set (`$nnUNet_raw/Dataset002_CSpineSeg_Gold/imagesTs/`) for all three models and
evaluate against `labelsTs` (expert labels). This is the apples-to-apples comparison.

Also predict the silver model on `Dataset003_CSpineSeg_Silver/imagesTs/` and evaluate
against silver labels → Biased Ruler analysis.

---

## Design Notes

- **Why 288 gold vs 450 silver train cases?** Structural property of CSpineSeg — fewer cases were expert-annotated. Acknowledged as a limitation; not a choice. The sex-balance (50/50) and race/age stratification are held constant.
- **Same config for all three models** — use whatever `find_best_configuration` selects for Dataset001 for consistency. If it picks 2d, train only 2d for 002/003 to save compute.
- **Gold test set is the reference** — `split_v3_gold` test (76 cases, expert labels) is used for all three models. Silver test set (138 cases) is only used for the Biased Ruler analysis.
