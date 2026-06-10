# Fairness Runs

Catalog of fairness evaluation runs in `outputs/fairness/`. Each run is a timestamped directory produced by the `src.fairness.evaluate` + `src.fairness.analyze` pipeline.

> **⚠️ Definition change (2026-06-07).** DPD/DIR were migrated from a continuous
> **mean-ratio** to the canonical **binarized rate-based** definition (success =
> `score > threshold`; DIR = `min/max` success rate). See
> [`dpd-dir-redefinition.md`](dpd-dir-redefinition.md). Consequently:
> - **Runs 1–6 below report OLD mean-based DIR/DPD numbers** and are kept only for
>   provenance — **do not cite their DIR/DPD values.** Their per-case Dice/HD95/nDSC
>   distributions are unaffected (only the aggregation changed).
> - The authoritative current numbers are the binarized reruns **Run 7** (global)
>   and **Run 8** (biased ruler), which reuse the same per-case CSVs.
> - In particular the old HD95 mean-DIRs of 0.40–0.77 are an outlier artifact; the
>   binarized HD95 DIR is ~0.88–1.00 (Run 7).
> - The old mean-based Run 6 gold→silver "DIR widening" table is **invalid** under
>   the new definition (the silver ruler saturates → DIR ≡ 1.0); see Run 8.

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

**Status**: Mean-based DIR/DPD superseded by **Run 7** (binarized rerun on the same per-case CSVs). The qualitative verdict (no significant gaps) is unchanged; only the HD95 DIR numbers move materially (0.40–0.77 → ~0.88–1.00).

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

> **⚠️ Mean-based — superseded by Run 8 (below).** The DIR/DPD
> values and the entire "widening" table in this Run 6 block use the old mean-ratio
> definition. **Do not cite them.** Under the binarized definition the silver ruler
> saturates (DIR ≡ 1.0 at 0.8), so the widening column is mechanically −100% and
> meaningless; the real biased-ruler signal is in the continuous tests (Run 8).

**Macro Dice DIR by grouping (OLD mean-based — do not cite):**

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

### Run 7 — `20260607_173932` (global rerun, binarized DIR/DPD)

| | |
|---|---|
| **Path** | `outputs/fairness/fairness_global/20260607_173932/` |
| **Ruler** | `global` (Dataset001 predictions vs `labelsTs`) |
| **Test cases** | 228 |
| **Metrics** | Dice + HD95 + nDSC (9 score columns) |
| **Definition** | **Binarized rate-based** DIR/DPD (Dice/nDSC > 0.8, HD95 < 5 mm) |
| **FDR significant** | **0 / 63** |

Re-aggregation of Run 5's per-case CSVs under the binarized definition. **The
authoritative global baseline.** No re-prediction; only the DIR/DPD aggregation changed.

**Macro DIR at threshold (lower = more disparity; 1.0 = parity):**

| Grouping | Dice DIR | nDSC DIR | HD95 DIR |
|---|---|---|---|
| sex | 0.9912 | 0.9825 | 0.9907 |
| race_wb | 0.9760 | 0.9692 | 0.9702 |
| race_wbo | 0.9692 | 0.9692 | 0.9702 |
| race_wn | 0.9830 | 0.9762 | 0.9760 |
| age_3bin | 0.9880 | 0.9778 | 0.9789 |
| age_median | 0.9903 | 0.9817 | 0.9970 |
| ethnicity | 0.9858 | 0.9905 | 0.8792 |

All Dice/nDSC DIRs ≥ 0.969. All HD95 DIRs ≥ 0.97 **except ethnicity (0.879, n=12
Hispanic)** — and even that clears the four-fifths 0.80 line. The old "alarming" HD95
mean-DIRs of 0.40–0.77 (Runs 4–5) are gone: binarization is outlier-robust by
construction. 0/63 tests FDR-significant. Read `sensitivity_global.csv` for the
threshold sweep.

**Status**: Current global baseline. Supersedes Run 5 (same data, new aggregation).

---

### Run 8 — `20260607_210826` (biased ruler rerun, binarized DIR/DPD)

| | |
|---|---|
| **Path** | `outputs/fairness/fairness_biased_ruler/20260607_210826/` |
| **Rulers** | `gold` (Dataset001 vs expert labels, 76) · `silver` (Dataset001 vs Dataset002 predictions, same 76) |
| **Metrics** | Dice + HD95 + nDSC (9 score columns) |
| **Definition** | **Binarized rate-based** DIR/DPD |
| **FDR significant** | gold: **0 / 63** · silver: **11 / 63 (all age)** |

Re-aggregation of Run 6's per-case CSVs under the binarized definition. **Supersedes
Run 6.**

> **Note:** a sibling directory `fairness_biased_ruler/20260607_173749/` also exists on
> disk — it is a **gold-only partial** (crashed on a corrupt reused silver CSV, contains
> no `*_silver*` outputs). Ignore it; `…_210826` is the complete run.

The headline is no longer the DIR-widening table — it reverses what the
binarized definition can say here:

- **The silver ruler saturates.** Silver scores Dataset001 ≈ 0.97 Dice on all 76
  cases → zero failures at threshold 0.8 → **silver DIR ≡ 1.0000, DPD ≡ 0.0000 for
  all 7 groupings.** The sweep (`sensitivity_silver.csv`) stays pinned at 1.0 through
  0.85 and only cracks at 0.90. **⇒ The single-threshold gold→silver widening cell is
  mechanically −100% and meaningless — do not cite it.** (Itself a finding: the
  disparity hides entirely *above* the "good enough" bar.)
- **The biased-ruler signal lives in the continuous tests, not the binarized DIR,
  and it is age — on *both* rulers.** Silver ruler: **11/63 FDR-significant, all age**
  (`age_3bin`/`age_median`, Dice & nDSC; 60+ worst). Gold ruler: **0/63 FDR**, but the
  *same-direction* age trend is its strongest signal (60+ worst; `ndsc_*__age` raw
  p≈0.002–0.008, `dice_*__age` raw p≈0.015–0.030) — it just sits sub-FDR (p_fdr≈0.13).
  Sex, race, ethnicity show nothing on either ruler at any threshold.
- **So the silver ruler does not *manufacture* the age effect — it changes the noise
  floor.** Against gold the age gap is *larger* in magnitude (≈2.7 Dice pts: <40 0.925 →
  60+ 0.898) but noisier, so it misses FDR. Against silver it is tiny (≈0.6 pts: 0.980 →
  0.974) yet ultra-low-variance (DS001≈DS002 twins → everyone ~0.97), so it clears FDR.
  The *significance verdict* flips purely from the choice of reference label. Clinically
  negligible either way; methodologically the point of the experiment.
- **Independent corroboration from the encoder probe** (`demographic-probing.../findings.md`,
  2026-05-07): age is the one axis encoded in Dataset001's bottleneck *above* its random-init
  null (3-bin BA **0.642 vs 0.476**, +0.166 task contribution), while **race sits at the null**
  (0.491 vs 0.373) — matching the audit's age-yes / race-no pattern. And **sex is encoded most
  strongly of all (AUROC 0.957) yet shows no performance gap**, a live instance of "encoding ≠
  disparity" (Petersen et al.) — so the age trend reflects genuine difficulty in older spines,
  not mere decodability. Consistent with Parikh et al.'s age-bias work.

**Gold ruler macro Dice DIR / DPD at 0.8 (real structure; success = Dice > 0.8):**

| Grouping | DIR | DPD | worst < best |
|---|---|---|---|
| race_wbo | 0.9375 | 0.0625 | Black < Other |
| age_3bin | 0.9524 | 0.0476 | <40 < 40–60 |
| race_wb | 0.9555 | 0.0436 | Black < White |
| ethnicity | 0.9697 | 0.0303 | Non-Hispanic < Hispanic |
| race_wn | 0.9749 | 0.0246 | Non-White < White |
| age_median | 0.9978 | 0.0021 | <53 < ≥53 |
| sex | 1.0000 | 0.0000 | tied — equal rate (0.974) in both |

All gold-ruler DIRs clear 0.80. For silver, all seven are 1.0000 (saturated).

**Status**: Current biased-ruler result. Supersedes Run 6.

---

### Run 9 — `20260609_163752` (bias amplification, binarized DIR/DPD)

| | |
|---|---|
| **Path** | `outputs/fairness/fairness_bias_amplification/20260609_163752/` |
| **Rulers** | `mixed` (Dataset001) · `gold_trained` (Dataset002) · `silver_trained` (Dataset003), all on 76 gold test cases |
| **Metrics** | Dice + HD95 + nDSC (9 score columns) |
| **Definition** | **Binarized rate-based** DIR/DPD |
| **FDR significant** | mixed: **0 / 63** · gold_trained: **0 / 63** · silver_trained: **0 / 63** |

Submitted via `sed 's/TPLSTAGE/bias_amplification/g' jobs/fairness_analysis.sh | bsub`
(job 28620483, run time ~2h40m). Per-case CSVs: `outputs/eval_ds1_on_gold.csv`,
`outputs/eval_ds2_on_gold.csv`, `outputs/eval_ds3_on_gold.csv`.

**Key result: no bias amplification.** Silver-trained (DS003) DIRs match mixed-trained (DS001)
almost exactly across all groupings. Gold-trained (DS002) is occasionally *worse* on disc
fairness, not better.

**Macro Dice DIR — all three models:**

| Grouping | Mixed | Gold-trained | Silver-trained |
|---|---|---|---|
| sex | 1.0000 | 0.9459 | 1.0000 |
| race_wb | 0.9555 | 0.9938 | 0.9555 |
| race_wbo | 0.9375 | 0.9375 | 0.9375 |
| race_wn | 0.9749 | 0.9863 | 0.9749 |
| age_3bin | 0.9524 | 0.9560 | 0.9524 |
| age_median | 0.9978 | 0.9955 | 0.9978 |
| ethnicity | 0.9697 | 0.9545 | 0.9697 |

**Disc Dice DIR (most sensitive):**

| Grouping | Mixed | Gold-trained | Silver-trained |
|---|---|---|---|
| sex | 1.0000 | 0.9429 | 0.9444 |
| race_wb | 0.9464 | 0.8971 | 0.9464 |
| race_wbo | 0.8750 | **0.8125** | 0.8750 |
| race_wn | 0.9876 | 0.9601 | 0.9876 |
| age_3bin | 0.8764 | 0.8885 | 0.8764 |
| age_median | 0.9502 | 0.8959 | 0.9502 |
| ethnicity | 0.9242 | 0.8939 | 0.9242 |

Gold-trained race_wbo disc DIR (0.813) is the only value approaching but still above the
four-fifths 0.80 threshold. Mixed and silver-trained both sit at 0.875. All other DIRs
comfortably above 0.80.

**Interpretation:** Silver training labels do not amplify demographic bias — contrary to
the MAMA-MIA finding (Parikh et al., 66% gap widening). The likely explanation is that
silver labels in CSpineSeg are high-quality (Zhou et al. trained on expert labels, applied
to the same scanner/acquisition distribution), so the noise introduced is insufficient to
differentially harm any demographic group. The larger silver training set (450 vs 288 gold
cases) may also help. This null result is itself a finding worth reporting.

**Status**: Current bias amplification result.

---

## Future runs

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
