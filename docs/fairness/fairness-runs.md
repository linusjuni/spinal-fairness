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

## Future runs

### Biased ruler (requires Dataset002 predictions)

Evaluate Dataset001 predictions on the 76 gold test images against both gold labels and Dataset002-generated silver labels. Same model, same images, different reference -- any difference in DIR is a pure ruler effect.

### Bias amplification (requires Dataset002 + Dataset003 predictions)

Evaluate Dataset001 (mixed-trained), Dataset002 (gold-trained), and Dataset003 (silver-trained) on the 76 gold test images against gold labels. If Dataset003 shows wider demographic gaps than Dataset002, silver labels amplify bias through training.

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
