# 05 — Inference & Evaluation

## Step 1: Select the Best Configuration

After all folds of all configurations have finished training (and `--npz` was used),
run automated model selection:

```bash
nnUNetv2_find_best_configuration 1 -c 2d 3d_fullres
```

This compares cross-validation Dice scores across all configurations and writes:
- `$nnUNet_results/Dataset001_CSpineSeg/inference_instructions.txt` — copy-paste predict command
- `$nnUNet_results/Dataset001_CSpineSeg/inference_information.json` — per-configuration metrics

> **Caveat:** If ensembling is enabled (default), the command also evaluates combinations
> of configurations (e.g. `2d + 3d_fullres` ensemble). Ensembles almost always
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

Flags:
- `-f 0 1 2 3 4` — uses all 5 folds as an ensemble (recommended)
- `--save_probabilities` — saves `.npz` softmax outputs; only needed if ensembling across
  configurations afterwards (large files — omit if disk space is tight)

> **Caveat:** Input images in `imagesTs/` must follow the same naming convention and file
> format as `imagesTr/` (i.e. `cspine_NNNNNN_0000.nii.gz`). A format mismatch causes a
> silent empty prediction.

---

## Step 3: Apply Postprocessing

nnU-Net learns postprocessing rules during `find_best_configuration` (e.g. keep only the
largest connected component). Apply these to the raw predictions:

```bash
nnUNetv2_apply_postprocessing \
    -i $nnUNet_results/Dataset001_CSpineSeg/predictions_test \
    -o $nnUNet_results/Dataset001_CSpineSeg/predictions_test_pp \
    --pp_pkl_file $nnUNet_results/Dataset001_CSpineSeg/crossval_results_folds_0_1_2_3_4/postprocessing.pkl \
    -plans_json $nnUNet_preprocessed/Dataset001_CSpineSeg/nnUNetPlans.json \
    -dataset_json $nnUNet_raw/Dataset001_CSpineSeg/dataset.json
```

The exact path to `postprocessing.pkl` is printed by `find_best_configuration`. Use
what is printed there, not a hardcoded path.

---

## Step 4: Compute Segmentation Metrics

nnU-Net v2 includes an evaluation script that computes Dice and Hausdorff distance per
case and per label:

```bash
nnUNetv2_evaluate_folder \
    -ref $SOURCE_SEGMENTATION_DIR \
    -pred $nnUNet_results/Dataset001_CSpineSeg/predictions_test_pp \
    -djfile $nnUNet_raw/Dataset001_CSpineSeg/dataset.json \
    -pfile $nnUNet_preprocessed/Dataset001_CSpineSeg/nnUNetPlans.json
```

Where `$SOURCE_SEGMENTATION_DIR` contains the ground-truth test masks (from
`/work3/s225224/data/cspineseg/extracted/segmentation/`), renamed to match prediction
filenames.

Output: a `summary.json` with per-case and mean metrics for each label class.

---

## Step 5: Fairness Analysis

Join `summary.json` metrics with `split_v3.tsv` demographics:

```python
# Pseudocode — see src/fairness/ (to be implemented)
metrics = load_summary_json(...)          # case_id → {dice_vb, dice_disc, hd95_vb, ...}
split   = load_split_v3(...)             # case_id → {sex, race_bin, age_bin, ...}
df      = metrics.join(split, on="case_id")

# Compare across subgroups
for group_col in ["sex_bin", "race_bin", "age_bin"]:
    for metric in ["dice_vb", "dice_disc"]:
        run_mann_whitney(df, group_col, metric)
        plot_boxplot(df, group_col, metric)
```

**Key analysis priorities (from EDA findings):**

| Comparison | Primary metric | Known confounder |
|---|---|---|
| Female vs Male | Dice (both labels) | Males have ~25% larger structures → volume-adjust before attributing gaps |
| White vs Black | Dice disc (cleaner target) | Age skew → stratify or include as covariate |
| Age bins | Dice both labels | — |
| Gold vs Silver | Dice both labels | Pending author data |

---

## Caveats

- **Test set is sacred.** Never run `nnUNetv2_predict` on the test set until training and
  model selection are fully complete. Any peeking — even just running inference to check
  it works — invalidates the holdout.
- **Hausdorff distance sensitivity.** HD95 is sensitive to single outlier voxels. Report
  alongside Dice; treat HD95 anomalies as signals to inspect individual cases.
- **Wide-FOV outliers.** 22 exams with >15 components (mostly White females on Siemens
  3T) may inflate error metrics. Flag these in the fairness analysis — they are not
  necessarily a fairness signal.
- **Volume adjustment.** Raw Dice comparisons between male and female cases are confounded
  by ~25% anatomical size difference. Consider normalising metrics by vertebral body
  volume (mm³) before reporting group gaps.
- **Gold/Silver distinction pending.** Until the Duke authors provide expert-annotated
  case IDs, all labels are treated as equivalent. The fairness analysis must be re-run
  once the distinction is available — this is a primary research question of the project.
