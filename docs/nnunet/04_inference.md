# 04 — Inference & Evaluation

> **Status: Blocked on training completion.**

## Step 1: Find Best Configuration

After all 10 training jobs finish:

```bash
uv run nnUNetv2_find_best_configuration 1 -c 2d 3d_fullres
```

This compares all configs, tests pairwise ensembles, determines postprocessing rules, and writes `inference_instructions.txt` with exact commands to copy-paste.

## Step 2: Predict on Test Set

Use the command from `inference_instructions.txt`. General form:

```bash
uv run nnUNetv2_predict \
    -i $nnUNet_raw/Dataset001_CSpineSeg/imagesTs \
    -o $nnUNet_results/Dataset001_CSpineSeg/predictions_test \
    -d 1 -c 2d -f 0 1 2 3 4 \
    -p nnUNetResEncUNetLPlans \
    -tr nnUNetTrainerWandB
```

## Step 3: Apply Postprocessing

```bash
uv run nnUNetv2_apply_postprocessing \
    -i $nnUNet_results/Dataset001_CSpineSeg/predictions_test \
    -o $nnUNet_results/Dataset001_CSpineSeg/predictions_test_pp \
    -pp_pkl_file <path from inference_instructions.txt> \
    -plans_json $nnUNet_preprocessed/Dataset001_CSpineSeg/nnUNetPlans.json \
    -dataset_json $nnUNet_raw/Dataset001_CSpineSeg/dataset.json
```

## Step 4: Compute Metrics

```bash
uv run nnUNetv2_evaluate_folder \
    <ground_truth_dir> \
    $nnUNet_results/Dataset001_CSpineSeg/predictions_test_pp \
    -djfile $nnUNet_raw/Dataset001_CSpineSeg/dataset.json \
    -pfile $nnUNet_preprocessed/Dataset001_CSpineSeg/nnUNetPlans.json
```

Outputs `summary.json` with per-case Dice and IoU for each label. HD95 is **not** included by default — compute separately with `surface-distance` for the fairness analysis.

## Step 5: Fairness Analysis

Join `summary.json` metrics with `split_v3.tsv` demographics and compare across subgroups (sex, race, age). Implementation in `src/fairness/` (not yet built).

Key confounders to control for:
- **Sex**: males have ~25% larger structures (volume-adjust Dice)
- **Race x Age**: White patients skew older (stratify or use age as covariate)
- **Disc label** is the cleaner fairness target (uniform across race and age)

Published baseline (Zhai et al.): ensemble Dice 0.929 VB / 0.904 disc / 0.916 macro.
