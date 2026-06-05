# Fairness Runs

Catalog of fairness evaluation runs in `outputs/fairness/fairness/`. Each run is a timestamped directory produced by the `src.fairness.evaluate` + `src.fairness.analyze` pipeline.

## Run log

### Run 1 — `20260501_160650` (initial global, Dice only)

| | |
|---|---|
| **Ruler** | `all` (Dataset001 predictions vs `labelsTs`) |
| **Test cases** | 228 (114F / 114M, split_v3) |
| **Metrics** | Dice (dice_vb, dice_disc, dice_macro) |
| **Groupings** | All 7 (sex, race_wb, race_wbo, race_wn, age_3bin, age_median, ethnicity) |
| **Output files** | 38 (stats.json, 21 summary CSVs, fdr_all.csv, 9 violins, dir_bar, forest) |

First fairness run. Single ruler, Dice only. Established the baseline: no significant demographic gaps in Dice (DIR > 0.98 across all groupings, 0 FDR-significant tests).

**Status**: Superseded by Run 3 (multi-ruler) and Run 4 (added HD95).

---

### Run 2 — `20260501_230017` (aborted)

Empty `stats.json`. Likely a failed or interrupted test run. Can be deleted.

---

### Run 3 — `20260501_230048` (three-ruler comparison, Dice only)

| | |
|---|---|
| **Rulers** | `all` (228 cases), `gold` (76 gold-labeled test cases), `silver` (138 silver-labeled test cases) |
| **Metrics** | Dice |
| **Groupings** | All 7 |
| **Output files** | 93 (3x summary sets, 3x fdr, 3x dir_bar/forest, 7 comparison CSVs, cross_ruler_dir.png, 27 violins) |

First cross-ruler comparison. Evaluated Dataset001 predictions against three subsets of `labelsTs`: all, gold-only, silver-only. Includes `comparison_*.csv` tables and `cross_ruler_dir.png`.

**Important caveat**: This is NOT the biased ruler analysis. Gold and silver rulers here evaluate on *different images* (76 gold vs 138 silver), not the same images with different reference labels. True biased ruler analysis requires generating silver labels for the 76 gold test images via Dataset002 (see README in `src/fairness/`).

**Status**: Useful as preliminary cross-ruler exploration. Not suitable for formal biased ruler conclusions.

---

### Run 4 — `20260507_110329` (global, Dice + HD95)

| | |
|---|---|
| **Ruler** | `global` (Dataset001 predictions vs `labelsTs`) |
| **Test cases** | 228 |
| **Metrics** | Dice + HD95 (6 score columns: dice_vb, dice_disc, dice_macro, hd95_vb, hd95_disc, hd95_macro) |
| **Groupings** | All 7 |
| **FDR tests** | 44 (7 groupings x 2 metric families x ~3 metrics, some with 2-group and some 3-group tests) |
| **Output files** | 70 (stats.json, 42 summary CSVs, fdr_global.csv, 14 violins, dir_bar, forest) |

The current global baseline. Key findings:

**Dice**: All DIRs 0.98--1.00. No test significant after FDR (all p_fdr > 0.64). Effect sizes negligible (|r_rb| < 0.06 except ethnicity at 0.24 with n=12). OLS regression: R^2 = 0.013, no covariate significant.

**HD95**: DIRs look alarming (0.40--0.63) but are misleading. HD95 has heavy right skew -- means are dominated by outlier cases. Medians are nearly identical across groups (F: 0.344mm vs M: 0.391mm for hd95_macro). Mann-Whitney tests (rank-based) confirm no significant differences. 0 FDR-significant tests.

**Status**: Current global baseline. Will be superseded by Run 5 (adds nDSC).

---

### Run 5 — `20260516_125043` (global, Dice + HD95 + nDSC)

| | |
|---|---|
| **Ruler** | `global` (Dataset001 predictions vs `labelsTs`) |
| **Test cases** | 228 (114F / 114M, split_v3) |
| **Metrics** | Dice + HD95 + nDSC (9 score columns) |
| **Groupings** | All 7 |
| **FDR tests** | 63 (7 groupings x 3 metric families x 3 metrics) |
| **Output files** | 97 (stats.json, 63 summary CSVs, fdr_global.csv, 21 violins, dir_bar, forest) |

Adds nDSC (Raina et al. 2023) to Run 4. nDSC decorrelates Dice from reference volume by scaling the FP penalty with a dataset-level effective load parameter, addressing the concern that males have ~25% larger vertebral bodies.

**Dice**: Identical to Run 4. All DIRs 0.98–1.00. 0 FDR-significant tests. OLS R² = 0.013, no covariate significant.

**HD95**: Identical to Run 4. DIRs 0.40–0.77 (misleading — see [HD95 outlier analysis](hd95-outliers.md)). 0 FDR-significant tests via Mann-Whitney.

**nDSC**: All DIRs 0.98–1.00, matching Dice within 0.4 percentage points. Same best/worst groups as Dice for every grouping. Volume correction does not change the fairness picture. 0 FDR-significant tests (lowest p_fdr = 0.72).

**Status**: Current global baseline. Supersedes Run 4.

---

### Run 6 — `20260605_121654` (biased ruler, Dice + HD95 + nDSC)

| | |
|---|---|
| **Path** | `outputs/fairness/biased_ruler/20260605_121654/` |
| **Rulers** | `gold` (Dataset001 vs expert labels, 76 cases) · `silver` (Dataset001 vs Dataset002 predictions, same 76 cases) |
| **Model** | Dataset001 (mixed-trained) |
| **Test cases** | 76 (gold test set, split_v3_gold) |
| **Metrics** | Dice + HD95 + nDSC (9 score columns) |
| **Groupings** | All 7 |
| **FDR significant** | gold: 0 · silver: 11 |

The biased ruler experiment: same model, same 76 images, two different reference labels. Ruler A is expert annotations; Ruler B is Dataset002's predictions on those same images (the "generated silver" ruler, mirroring how CSpineSeg's original silver labels were created).

**Macro Dice DIR by grouping:**

| Grouping | DIR (gold ruler) | DIR (silver ruler) | Widening | Direction |
|---|---|---|---|---|
| sex | 0.9914 | 0.9985 | −82.6% | narrowed |
| race_wb | 0.9704 | 0.9948 | −82.4% | narrowed |
| race_wbo | 0.9568 | 0.9905 | −78.0% | narrowed |
| race_wn | 0.9838 | 0.9977 | −85.8% | narrowed |
| age_3bin | 0.9932 | 0.9931 | +1.7% | widened |
| age_median | 0.9978 | 0.9964 | +60.0% | widened |
| ethnicity | 0.9881 | 0.9964 | −69.9% | narrowed |

Negative widening means the silver ruler makes the gap appear smaller (masks it). All race and sex groupings narrow by 70–86%; age groupings are near-neutral or slightly widen. All DIRs on both rulers remain well above the four-fifths threshold (0.80).

**Gold ruler macro Dice DPD (worst gaps):**
- race_wb: DPD=0.027 (White 0.901 > Black 0.875)
- race_wbo: DPD=0.039 (Other 0.914 > Black 0.875)
- sex: DPD=0.008 (Female 0.901 > Male 0.893)

**Silver ruler macro Dice DPD (same groupings):**
- race_wb: DPD=0.005 (White 0.974 > Black 0.969)
- race_wbo: DPD=0.009 (Other 0.978 > Black 0.969)
- sex: DPD=0.002 (Male 0.974 > Female 0.973)

**Commands used:**
```bash
source .env
uv run -m src.fairness.evaluate \
    --predictions "${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp" \
    --references  "${nnUNet_raw}/Dataset001_CSpineSeg/labelsTs_gold" \
    --mapping     "${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json" \
    --output      "${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp/eval_ruler_gold.csv" \
    --metrics dice hd95 ndsc --workers 24

uv run -m src.fairness.evaluate \
    --predictions "${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp" \
    --references  "${nnUNet_results}/Dataset002_CSpineSeg_Gold/predictions_test_pp" \
    --mapping     "${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json" \
    --output      "${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp/eval_ruler_silver.csv" \
    --metrics dice hd95 ndsc --workers 24

uv run -m src.fairness.analyze \
    --evaluation-csvs \
        "${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp/eval_ruler_gold.csv" \
        "${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp/eval_ruler_silver.csv" \
    --ruler-labels gold silver \
    --mapping "${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json" \
    --report-name biased_ruler
```

---

## Future runs

### Run 7 — Mixed vs gold + Bias amplification (blocked until Dataset003 completes)

Evaluate Dataset001, Dataset002, and Dataset003 on the 76 gold test images against gold labels.
Compare fairness gaps (DPD, DIR):
- Dataset002 vs Dataset003 → does silver training widen gaps? (bias amplification)
- Dataset001 vs Dataset002 → does mixing silver into training hurt fairness? (mixed vs gold)

## Outputs per run

Each timestamped directory contains:

| File | Description |
|---|---|
| `stats.json` | All fairness metrics, statistical tests, bootstrap CIs, permutation tests, OLS regression |
| `summary_{ruler}__{metric}__{grouping}.csv` | Per-group descriptive statistics (n, mean, median, std, IQR) |
| `fdr_{ruler}.csv` | Raw and BH-FDR corrected p-values for all tests in the run |
| `violin_{metric}_by_{group_col}.png` | Score distributions by demographic group |
| `dir_bar_{ruler}.png` | DIR bar chart with four-fifths rule (0.8) threshold |
| `forest_{ruler}.png` | Bootstrap DIR point estimates with 95% CIs |
| `cross_ruler_dir.png` | (Multi-ruler only) Grouped bar chart comparing DIR across rulers |
| `comparison_{grouping}.csv` | (Multi-ruler only) Side-by-side fairness gap comparison |
