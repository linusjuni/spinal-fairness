# DPD / DIR Redefinition — Migration Note (2026-06-07)

Handoff note for anyone (human or LLM) picking up the fairness code. It records
exactly how DPD and DIR were defined **before**, how they are defined **now**,
why it changed, and what still needs to happen.

## TL;DR

The Disparate Impact Ratio (DIR) and Demographic Parity Difference (DPD) were
changed from a **continuous mean-ratio** to the **canonical binarized rate-based**
definition used by Parikh et al. (MAMA-MIA), Fairlearn, and the EEOC four-fifths
rule. This makes our numbers comparable to the paper we extend and makes the 0.80
four-fifths threshold legitimate. Per-case metric CSVs are unaffected; only the
aggregation changed, so the analyses must be **re-aggregated** (not re-predicted).

---

## BEFORE (continuous, mean-based)

DPD/DIR operated directly on the **mean score** per group (raw Dice, HD95, nDSC).

```
DIR = mean(worst_group) / mean(best_group)
DPD = mean(best_group) − mean(worst_group)
```

- Implemented in `src/fairness/metrics.py` via a `_group_means()` helper.
- `fairness_gap()` returned keys including `best_mean` / `worst_mean`.
- `disparate_impact_ratio(df, score_col, group_col)` — no threshold argument.
- "best"/"worst" were decided by mean magnitude; for HD95 (lower-is-better) the
  convention was baked into the docstring, not a parameter.

### Problems with the old definition
1. **Not comparable** to Parikh et al. — their DIR/DPD binarize a beneficial
   outcome (Dice > 0.8) and compare **success rates**, not means.
2. **Four-fifths misuse** — the 0.80 threshold is only defined for selection-rate
   ratios. Applying it to a continuous mean-ratio was not valid.
3. **HD95 fragility** — HD95 is heavily right-skewed and unbounded, so a single
   outlier case dominates the group mean and tanks the mean-based DIR. This is
   the entire subject of `hd95-outliers.md` (a mean-vs-median DIR workaround).

---

## AFTER (binarized, rate-based)

Each case is binarized into a **beneficial outcome** ("success"); a per-group
**success rate** `P(ŷ=1 | group)` is computed; then:

```
DPD = max(rate) − min(rate)        # absolute gap in success rate
DIR = min(rate) / max(rate)        # relative gap; 1.0 = parity
```

- Beneficial outcome: `score > threshold` when `higher_is_better` (Dice, nDSC),
  else `score < threshold` (HD95, lower-is-better, in mm).
- Default thresholds: **Dice 0.8, nDSC 0.8, HD95 5 mm** — all configurable.
- The 0.80 four-fifths rule now **legitimately** applies (DIR is a rate ratio).
- HD95 binarization is **outlier-robust by construction** (a 54 mm case is one
  "failure", not a mean-wrecking magnitude) → supersedes the `hd95-outliers.md`
  mean-vs-median workaround.
- A **sensitivity sweep** is emitted per run (`sensitivity_<ruler>.csv`) because
  our macro Dice (~0.89) sits above 0.8, so a single cutoff can be near-degenerate
  (success rates near 1.0). Confirms the verdict is threshold-robust.

> Four-fifths caveat ("portability trap"): the 0.80 rule comes from US employment
> law; Fairlearn notes it has no validity outside that context. We adopt it as the
> field convention + for comparability with Parikh et al., not as a legal claim.

---

## Code changes

### `src/fairness/metrics.py`
| Before | After |
|---|---|
| `_group_means(df, score, group)` | `_group_rates(df, score, group, threshold, higher_is_better)` |
| `disparate_impact_ratio(df, score, group)` → mean ratio | `disparate_impact_ratio(df, score, group, threshold=0.8, higher_is_better=True)` → `min/max` rate |
| `demographic_parity_difference(df, score, group)` → mean diff | same signature + `threshold`/`higher_is_better` → `max−min` rate |
| `fairness_gap(...)` keys `best_mean`/`worst_mean` | keys `best_rate`/`worst_rate`, plus `threshold`, `higher_is_better`; also fixed an inline-DIR duplication bug |
| — | **new** `dir_sensitivity(df, score, group, thresholds, higher_is_better)` → DataFrame of DIR/DPD per threshold |
| `compare_fairness_gaps` read `best_mean`/`worst_mean` | reads `best_rate`/`worst_rate` |

`bootstrap_ci` / `permutation_test` bodies are unchanged — they take a
`metric_fn(df, score, group)`; the threshold + direction are bound at the call
site via `functools.partial`, so CIs and permutation p-values recompute under the
new DIR automatically.

### `src/fairness/analyze.py`
- New config `DEFAULT_THRESHOLDS = {dice:0.8, ndsc:0.8, hd95:5.0}`, sweep grids,
  and helpers `_beneficial_spec(score_col)` → `(threshold, higher_is_better)` and
  `_sweep_for(score_col, ...)`.
- Call sites pass `threshold`/`higher_is_better` into `fairness_gap`, and a
  `functools.partial(disparate_impact_ratio, threshold=…, higher_is_better=…)`
  into `bootstrap_ci`/`permutation_test`.
- Writes `sensitivity_<ruler>.csv` per ruler.
- New CLI flags: `--dice-threshold`, `--ndsc-threshold`, `--hd95-threshold`,
  `--sweep-higher`, `--sweep-hd95`.

### Unchanged on purpose
- `src/fairness/evaluate.py` (per-case Dice/HD95/nDSC) — **identical output**.
- `src/fairness/plots.py` — only reads `gap["dir"]`; the 0.80 reference line is
  now legitimately meaningful.
- Mann-Whitney / Kruskal-Wallis / OLS — operate on continuous scores; unchanged.
- `group_summary` still reports continuous mean/median/IQR as descriptive stats.

### Robustness hardening (2026-06-09, not load-bearing for the result)
Added after the first biased-ruler rerun crashed. The actual cause was a corrupt
cached CSV (above), not these — they never fired (0 skips) — but they're cheap
insurance and stay in:
- `src/eda/stats.py` — `mann_whitney_result` / `kruskal_result` short-circuit to
  `p=1.0` when input is all-identical (scipy raises on this in some versions;
  ours warns). Relevant to the saturated silver ruler.
- `src/fairness/analyze.py` — the per-`(score, grouping)` body is wrapped in
  `try/except`-continue (mirrors the existing OLS guard), so one degenerate metric
  can't abort a whole ruler and silently drop the silver ruler + widening.

---

## Documentation changes (definitional only)
- `docs/statistical-testing/statistical-testing.md` — DPD/DIR section rewritten to
  the binarized definition + thresholds + sweep + four-fifths caveat.
- `src/fairness/README.md` — updated function signatures + a "How DPD/DIR are
  defined" note.
- `docs/fairness/hd95-outliers.md` — **superseded banner** added (binarized HD95
  DIR is outlier-robust; the mean-vs-median analysis is no longer needed).
- **Not touched** (results docs, by decision): `fairness-runs.md` (Run 1–6
  numbers), `06_gold_silver_training.md`. These still quote OLD mean-based DIR/DPD
  numbers and must be regenerated from the rerun before being trusted/cited.

---

## Job scripts
- **Removed** the monolithic sequential `jobs/fairness_suite.sh`.
- **`jobs/fairness_analysis.sh`** — `TPLSTAGE`-templated single-analysis `hpc` job
  (`case` on `STAGE ∈ {global, biased_ruler, bias_amplification}`). Each stage is
  input-guarded (`require_paths`) and skips cleanly (`exit 0`) if predictions/
  references are missing. `evaluate` is reuse-if-exists (`ensure_eval`).
- **`jobs/submit_fairness.sh`** — submits one job per stage so they run in
  parallel: `sed "s/TPLSTAGE/${stage}/g" … | bsub`.
- Run with `bash jobs/submit_fairness.sh [stage ...]`.

---

## What must be rerun (status)

Per-case CSVs do NOT change → **no re-prediction, no `evaluate` recompute** (it is
reused). Only re-run `src.fairness.analyze` (via the jobs above):

| Analysis | Old run (mean-based) | Status under new definition |
|---|---|---|
| Global audit (Dataset001, 228) | Run 5 `20260516_125043` | **DONE** `fairness_global/20260607_173932` — 0/63 FDR-sig, all DIRs ≥ 0.969 |
| Biased ruler (gold vs gen-silver, 76) | Run 6 `20260605_121654` | **DONE** `fairness_biased_ruler/20260607_210826` — gold 0/63, silver 11/63 FDR-sig (all age) |
| Bias amplification (DS1/2/3 vs gold, 76) | never run | **DONE** `fairness_bias_amplification/20260609_163752` (Run 9) — 0/63 FDR all three models; silver-trained DIRs ≈ mixed; **no amplification** |
| HD95 outlier / median-DIR analysis | `hd95-outliers.md` | **superseded** — retire/re-derive, don't simply rerun |

Old mean-based runs (`outputs/fairness/fairness/`, `…/biased_ruler/`) are moved to
`outputs/fairness/archive/` (superseded, do not cite). The first biased-ruler
rerun `fairness_biased_ruler/20260607_173749/` is a **gold-only partial** (crashed
on a corrupt reused silver CSV — see gotcha below); the good run is `…_210826`.

Results docs updated (2026-06-09) with the new numbers + silver-saturation framing:
`fairness-runs.md` (banner + Run 7/Run 8 entries, Run 1–6 marked mean-based/superseded),
`06_gold_silver_training.md` (status block + biased-ruler subsection), `docs/README.md`
(index entry for this note + corrected descriptions). `hd95-outliers.md` already carries
the superseded banner. Also fixed `src/fairness/README.md`, which still used
`--ruler-labels gold generated_silver` (silently skips the widening) → `gold silver`.

Bias-amplification run (Run 9) complete as of 2026-06-10 — see `fairness-runs.md` Run 9 and
`06_gold_silver_training.md`. No further reruns outstanding for the binarized migration.

---

## Update (2026-06-09) — rerun complete + what binarization does to the silver ruler

The reruns landed. The global + gold-ruler results behave exactly as intended
(real DIR structure at 0.8, four-fifths legitimately applies). **The silver ruler
is the surprise, and it's a genuine limitation of the binarized definition for
this comparison:**

- **The silver ruler saturates completely.** Silver scores DS001 against DS002's
  predictions → Dice ≈ 0.97 for *every* one of the 76 cases. At threshold 0.8
  there are **zero failures in any group**, so **silver DIR = 1.0000 and DPD =
  0.0000 for all 7 groupings**. The sweep confirms it: silver DIR stays pinned at
  1.0 through 0.85 and only cracks at 0.90.
- **⇒ The single-threshold gold→silver "DIR widening" headline is meaningless
  here.** It prints "narrowed −100%" for everything, purely because the silver
  side is mechanically pinned at 1.0. **Do not cite the 0.8 widening table.** (The
  old mean-based widening was actually *more* informative for the silver ruler.)
- **The biased-ruler signal lives in the continuous tests, not the binarized DIR —
  and it is age, on *both* rulers.** Silver ruler: **11/63 FDR-significant, all age**
  (`age_3bin`/`age_median`, Dice & nDSC; 60+ worst). Gold ruler: **0/63 FDR**, but the
  *same-direction* age trend is its strongest signal too (60+ worst; nDSC age raw
  p≈0.002–0.008, Dice age raw p≈0.015–0.030) — it just lands sub-FDR (p_fdr≈0.13).
  Sex/race/ethnicity show nothing on either ruler.
- **The silver ruler does not *manufacture* the age effect — it lowers the noise
  floor.** Against gold the gap is *bigger* (≈2.7 Dice pts: <40 0.925 → 60+ 0.898) but
  noisy → misses FDR. Against silver it is tiny (≈0.6 pts: 0.980 → 0.974) yet
  ultra-low-variance (DS001≈DS002 are near-twins → everyone ~0.97) → clears FDR. The
  *significance verdict* flips from the choice of reference label, not from the effect
  appearing or disappearing. Clinically negligible either way (~0.6 pts in a ~0.97
  regime); methodologically it *is* the experiment's point — consistent with Parikh et
  al.'s age-bias work.
- **Independent corroboration (encoder probe, `demographic-probing-of-medical-image-encoders/findings.md`):**
  age is the one attribute encoded in Dataset001's bottleneck above its random-init null
  (3-bin BA 0.642 vs 0.476, +0.166), while race sits at the null (0.491 vs 0.373) —
  mirroring the audit's age-yes / race-no pattern. Sex is encoded *most* strongly
  (AUROC 0.957) yet has no performance gap → encoding ≠ disparity (Petersen et al.), so
  the age trend reflects genuine difficulty in older spines, not mere decodability.

**Recommendation for the write-up:** keep the binarized DIR as the headline for
global + gold (it works). For the silver ruler, report that it saturates at 0.8
(DIR = 1.0 — itself a point: the disparity hides entirely *above* the "good
enough" bar), read its structure off the sweep, and rest the biased-ruler
*significance* claim on the continuous Mann-Whitney/Kruskal + nDSC tests. We are
**not** abandoning the new DIR — only the single-cutoff silver widening cell.

---

## Gotchas for the next person
- **Near-degenerate threshold**: at 0.8 with Dice ~0.89, success rates are near
  1.0, so DIR sits near 1.0 and its bootstrap CI / permutation p-value can be
  unstable (few "failures"). Read `sensitivity_*.csv`, don't trust a single cutoff.
  **Extreme case — the silver ruler is fully saturated** (Dice ~0.97 → zero
  failures → DIR ≡ 1.0); its `.err` is full of scipy `DegenerateDataWarning` (BCa
  CI uncomputable). That's expected, not a regression — see the 2026-06-09 update.
- **Corrupt reused eval CSV (cost us a whole run)**: `ensure_eval` reuses
  `outputs/eval_*.csv` if the file exists. A job killed mid-write leaves a
  truncated CSV that `analyze` then chokes on every rerun. The fix that worked:
  `rm -f outputs/eval_ruler_silver.csv` before resubmitting to force a fresh
  evaluate. If a stage fails mysteriously, suspect the cached CSV first.
- **Capture tracebacks**: the job now sets `PYTHONUNBUFFERED=1` and runs
  `uv run python -u -m …`. The first failed rerun lost its traceback entirely
  (empty `.err`) because output was buffered when the process died — don't revert
  this.
- **Biased-ruler labels must be `gold` and `silver`**: `analyze` only computes the
  DIR-widening (the headline) when both literal labels are present. Do not rename
  the silver ruler to `generated_silver`.
- **Direction**: HD95 is lower-is-better; `_beneficial_spec` flips it
  automatically via `score_col.startswith("hd95")`. Any new lower-is-better metric
  needs adding there.
- **Old vs new numbers will differ a lot for HD95** specifically (old mean-DIR
  0.40–0.77 → new rate-DIR ~0.95), because the outlier sensitivity is gone. This
  is expected, not a regression.
