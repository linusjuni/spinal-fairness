# src/fairness/

Fairness evaluation module for CSpineSeg segmentation. Computes per-case segmentation metrics from NIfTI predictions, applies demographic groupings, and produces fairness analysis (DIR, DPD, statistical tests, bootstrap CIs).

## Tutorials

Run the full pipeline on one fold's validation predictions to verify everything works end-to-end:

```bash
# 1. Evaluate: NIfTI predictions → per-case CSV (Dice only, ~2 min)
uv run -m src.fairness.evaluate \
    --predictions ${nnUNet_results}/Dataset001_CSpineSeg/nnUNetTrainerWandB__nnUNetResEncUNetLPlans__2d/fold_0/validation \
    --references  ${nnUNet_raw}/Dataset001_CSpineSeg/labelsTr \
    --mapping     ${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json \
    --output      /tmp/eval_fold0.csv \
    --split       train

# 2. Analyze: CSV + demographics → fairness report
uv run -m src.fairness.analyze \
    --evaluation-csvs /tmp/eval_fold0.csv \
    --ruler-labels    val_fold0 \
    --mapping         ${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json
```

Output lands in `outputs/fairness/fairness/<timestamp>/` with:
- `stats.json` — all fairness metrics, statistical tests, bootstrap CIs
- `summary_*.csv` — per-group descriptive statistics for every (score, grouping) pair
- `fdr_*.csv` — BH-FDR corrected p-values
- `violin_*.png` — score distributions by demographic group
- `dir_bar_*.png` — DIR bar chart with four-fifths rule threshold

## How-to guides

### Global fairness audit (Dataset001 on full test set)

```bash
uv run -m src.fairness.evaluate \
    --predictions ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp \
    --references  ${nnUNet_raw}/Dataset001_CSpineSeg/labelsTs \
    --mapping     ${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json \
    --output      outputs/eval_global.csv

uv run -m src.fairness.analyze \
    --evaluation-csvs outputs/eval_global.csv \
    --ruler-labels    global \
    --mapping         ${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json
```

### Biased ruler analysis (gold labels vs generated silver)

CSpineSeg images have either gold or silver labels, not both (unlike MAMA-MIA). The adapted approach: use Dataset002 (gold-trained) to generate predictions on the 76 gold test images — those serve as "generated silver labels." We then evaluate Dataset001 against both rulers on the same images.

```bash
# Ruler 1: evaluate Dataset001 against gold labels → true performance
uv run -m src.fairness.evaluate \
    --predictions ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp \
    --references  ${nnUNet_raw}/Dataset001_CSpineSeg/labelsTs_gold \
    --mapping     ${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json \
    --output      outputs/eval_ruler_gold.csv

# Ruler 2: evaluate Dataset001 against Dataset002's predictions (generated silver)
uv run -m src.fairness.evaluate \
    --predictions ${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp \
    --references  ${nnUNet_results}/Dataset002_CSpineSeg_Gold/predictions_test_pp \
    --mapping     ${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json \
    --output      outputs/eval_ruler_silver.csv

# Compare fairness gaps between the two rulers
uv run -m src.fairness.analyze \
    --evaluation-csvs outputs/eval_ruler_gold.csv outputs/eval_ruler_silver.csv \
    --ruler-labels    gold silver \
    --mapping         ${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json
```

> The labels must be literally `gold` and `silver` — `analyze.py` only computes the
> gold→silver DIR-widening when both of those exact labels are present. Do not rename
> the second ruler to `generated_silver`.

Any difference in the fairness gap between the two evaluations is the pure ruler effect: same model, same images, different reference labels. Dataset001 and Dataset002 are independently trained, so Dice != 1.

### Include HD95 and nDSC

```bash
uv run -m src.fairness.evaluate \
    --predictions ... --references ... --mapping ... --output ... \
    --metrics dice hd95 ndsc \
    --workers 24
```

HD95 uses `surface-distance` and is significantly slower (~10x). nDSC (Normalized Dice, Raina et al. 2023) decorrelates Dice from reference volume — it scales the FP penalty by the ratio of the case's class load to the dataset mean, so that cases with smaller structures aren't penalized more for the same absolute error. The effective load parameter is computed automatically from all evaluated cases. Use `--workers N` to parallelize across cases (each case is independent).

### Bias amplification (Dataset002 vs Dataset003)

After training Dataset002 (gold-only) and Dataset003 (silver-only), predict on the gold test set (76 cases) and evaluate both against gold labels. If Dataset003 shows wider demographic gaps, silver labels amplify bias through training.

```bash
# Evaluate each model against gold test labels
for DS in 1 2 3; do
    uv run -m src.fairness.evaluate \
        --predictions ${nnUNet_results}/Dataset00${DS}_CSpineSeg*/predictions_test_pp \
        --references  ${nnUNet_raw}/Dataset001_CSpineSeg/labelsTs_gold \
        --mapping     ${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json \
        --output      outputs/eval_ds${DS}_on_gold.csv
done

# Compare all three
uv run -m src.fairness.analyze \
    --evaluation-csvs outputs/eval_ds1_on_gold.csv outputs/eval_ds2_on_gold.csv outputs/eval_ds3_on_gold.csv \
    --ruler-labels    mixed gold_trained silver_trained \
    --mapping         ${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json
```

## Reference

### Module structure

| File | Purpose |
|---|---|
| `__init__.py` | Label constants: `LABEL_VERTEBRAL_BODY=1`, `LABEL_DISC=2`, `LABELS={1: "vb", 2: "disc"}` |
| `evaluate.py` | NIfTI I/O. Computes per-case Dice, HD95, and nDSC per label. Emits CSV. |
| `metrics.py` | Pure functions. All take `(df, score_col, group_col)`. No I/O. |
| `plots.py` | Visualization functions. Each takes data + `EDAReport`. |
| `analyze.py` | Orchestrator. Loads CSVs, joins demographics, calls metrics + plots. |

### evaluate.py

| Function | Signature | Returns |
|---|---|---|
| `dice_coefficient` | `(pred, ref, label)` | `float`. NaN if both empty, 0.0 if one empty. |
| `hausdorff_95` | `(pred, ref, label, spacing)` | `float` in mm. NaN if both empty, inf if one empty. |
| `_segmentation_counts` | `(pred, ref, label)` | `dict` with tp, fp, fn, ref_vol. Used internally by nDSC. |
| `_compute_ndsc` | `(df)` | `pl.DataFrame` with ndsc_{label} columns. Two-pass: computes effective_load per label from the dataset, then nDSC per case. |
| `evaluate_case` | `(pred_path, ref_path, case_id, series_submitter_id, metrics={"dice"})` | `dict` with case_id, series_submitter_id, dice/hd95/ndsc per label |
| `evaluate_folder` | `(pred_dir, ref_dir, mapping, output_path=None, metrics={"dice"}, split="test")` | `pl.DataFrame` |

### metrics.py

| Function | Signature | Returns |
|---|---|---|
| `group_summary` | `(df, score_col, group_col)` | `pl.DataFrame` — n, mean, median, std, q25, q75, iqr per group |
| `disparate_impact_ratio` | `(df, score_col, group_col, threshold=0.8, higher_is_better=True)` | `float` — min/max group success rate. <0.8 is four-fifths rule violation. |
| `demographic_parity_difference` | `(df, score_col, group_col, threshold=0.8, higher_is_better=True)` | `float` — max - min group success rate |
| `fairness_gap` | `(df, score_col, group_col, threshold=0.8, higher_is_better=True)` | `dict` — DIR, DPD, best/worst group identities + success rates |
| `mann_whitney_test` | `(df, score_col, group_col)` | `dict` — U, p, rank-biserial r. For 2 groups. |
| `kruskal_wallis_test` | `(df, score_col, group_col)` | `dict` — H, p, epsilon-squared, Dunn's post-hoc. For 3+ groups. |
| `apply_fdr` | `(p_values, method="fdr_bh")` | `list[float]` — BH-corrected p-values |
| `ols_regression` | `(df, score_col, covariates)` | `dict` — coefficients, R-squared, F-stat, CIs |
| `bootstrap_ci` | `(df, score_col, group_col, metric_fn, ...)` | `dict` — BCa bootstrap CI for DIR or DPD |
| `permutation_test` | `(df, score_col, group_col, metric_fn, ...)` | `dict` — observed value, empirical p-value |
| `dir_widening` | `(dir_gold, dir_silver)` | `dict` — widening %, direction |
| `dir_sensitivity` | `(df, score_col, group_col, thresholds, higher_is_better=True)` | `pl.DataFrame` — DIR/DPD across a threshold sweep |
| `compare_fairness_gaps` | `(gaps, labels)` | `pl.DataFrame` — side-by-side comparison table |

### plots.py

| Function | What it draws |
|---|---|
| `violin_by_group` | Violin plot of score by demographic group |
| `dir_bar_chart` | Horizontal bar chart of DIR with 0.8 threshold line |
| `cross_ruler_dir` | Grouped bar chart comparing DIR across rulers |
| `bootstrap_forest` | Forest plot with point estimates + CI error bars |

### Demographic groupings (from `src/data/groups.py`)

The analyzer applies 7 grouping strategies: sex (binary), race white-vs-black, race 3-way, race white-vs-nonwhite, age 3-bin, age median-split, and ethnicity hispanic-vs-not. Each drops unmapped rows via `GroupingSpec.apply()`.

## Explanation

### How DPD and DIR are defined (binarized, Parikh et al.)

DPD and DIR follow the canonical fairness definition (Parikh et al. MAMA-MIA;
Fairlearn; EEOC four-fifths rule), **not** a continuous mean ratio. Each case is
binarized into a *beneficial outcome* ("success") at a threshold, a per-group
success rate is computed, then `DPD = max(rate) - min(rate)` and
`DIR = min(rate) / max(rate)`. Success is `score > threshold` for Dice/nDSC
(higher-is-better) and `score < threshold` for HD95 (lower-is-better, mm).
Defaults: Dice/nDSC `0.8`, HD95 `5 mm` (override via `--dice/-ndsc/-hd95-threshold`).
Because DIR is a rate ratio, the 0.80 four-fifths rule legitimately applies.

Each run also writes `sensitivity_{ruler}.csv` — DIR/DPD across a threshold sweep
(`--sweep-higher`, `--sweep-hd95`) — since our Dice (~0.89) sits above 0.8 and a
single cutoff could be near-degenerate. See `docs/statistical-testing/` for the
full definition and the Fairlearn "portability trap" caveat on the threshold.

### Biased ruler: why generated silver instead of actual silver labels?

In MAMA-MIA, every image had both a gold and a silver label, so the ruler comparison was direct. CSpineSeg images have either gold or silver — not both. Evaluating against `labelsTs_gold/` vs `labelsTs_silver/` would compare different images (76 gold vs 138 silver), not different rulers on the same images.

The adapted approach: use Dataset002 (gold-trained) to generate predictions on the 76 gold test images as "generated silver labels." Now both rulers exist for the same images — gold expert labels and model-generated labels. We evaluate Dataset001 against both rulers; any difference in the observed fairness gap is the pure ruler effect: same model, same images, different reference. This follows the methodology clarified with Aditya (meeting 2026-05-01).

### Why three models?

Training on silver labels may not just affect measurement — it can amplify bias through the training loop. Dataset002 (gold-trained) and Dataset003 (silver-trained) are both evaluated on the same 76 gold test cases. If Dataset003 shows wider demographic gaps, silver labels amplify bias through training. Parikh et al. found 66% DIR widening in MAMA-MIA. Dataset001 (mixed) serves as the production-realistic baseline.

### Why NaN for both-empty masks?

When neither prediction nor reference contains a label, there is nothing to measure. Returning 0.0 would drag down group means; returning 1.0 would inflate them. NaN (skip) matches the nnU-Net convention and lets aggregations use `nanmean` to exclude these cases transparently.

### Why Dice defaults, HD95 and nDSC opt-in?

Dice is O(n) on voxel counts. HD95 requires computing surface distances between 3D meshes — roughly 10x slower per case. nDSC has the same computational cost as Dice (it reuses voxel counts) but requires a two-pass computation across all cases to derive the effective load. For rapid iteration, Dice-only evaluation completes in minutes; HD95 and nDSC can be added when needed for final analysis.

### Why nDSC alongside Dice?

Standard Dice has two properties that can confound fairness analysis:

1. **Volume dependence**: The same absolute segmentation error produces a larger Dice drop for smaller structures. If one demographic group has systematically smaller structures, Dice will make them look worse even with identical segmentation quality.
2. **Erosion/dilation asymmetry**: Undersegmentation (erosion) removes TP from both numerator and denominator, while oversegmentation (dilation) only adds FP to the denominator. Equal-magnitude errors produce unequal Dice changes.

nDSC (Raina et al. 2023) addresses this by scaling the FP penalty by `kappa = (1-r) * |R| / (r * (N-|R|))`, where `r` is the mean class load across the dataset. When a case's class load equals the dataset mean, nDSC = Dice. When it's smaller, FP is penalized less (boosting the score); when larger, FP is penalized more (reducing the score). This decorrelates the metric from reference volume.

### Volume adjustment via OLS

Males have ~25% larger vertebral bodies than females. Raw Dice differences between sexes may reflect anatomy, not model bias. The OLS regression controls for segmentation volume (`volume_mm3_vertebral_body`) as a covariate, testing whether demographic effects on Dice persist after accounting for structure size.
