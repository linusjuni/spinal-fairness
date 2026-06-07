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
| Global audit (Dataset001, 228) | Run 5 `20260516_125043` | **rerun** (`fair_global`) |
| Biased ruler (gold vs gen-silver, 76) | Run 6 `20260605_121654` | **rerun** (`fair_biased_ruler`) |
| Bias amplification (DS1/2/3 vs gold, 76) | never run | **blocked** — Dataset003 7/10 trained, then must predict on the 76 gold test images |
| HD95 outlier / median-DIR analysis | `hd95-outliers.md` | **superseded** — retire/re-derive, don't simply rerun |

After rerun, update the results docs (`fairness-runs.md`, `06_…`, retire
`hd95-outliers.md`) with the new numbers.

---

## Gotchas for the next person
- **Near-degenerate threshold**: at 0.8 with Dice ~0.89, success rates are near
  1.0, so DIR sits near 1.0 and its bootstrap CI / permutation p-value can be
  unstable (few "failures"). Read `sensitivity_*.csv`, don't trust a single cutoff.
- **Biased-ruler labels must be `gold` and `silver`**: `analyze` only computes the
  DIR-widening (the headline) when both literal labels are present. Do not rename
  the silver ruler to `generated_silver`.
- **Direction**: HD95 is lower-is-better; `_beneficial_spec` flips it
  automatically via `score_col.startswith("hd95")`. Any new lower-is-better metric
  needs adding there.
- **Old vs new numbers will differ a lot for HD95** specifically (old mean-DIR
  0.40–0.77 → new rate-DIR ~0.95), because the outlier sensitivity is gone. This
  is expected, not a regression.
