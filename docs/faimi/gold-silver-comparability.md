# Gold vs. Silver Pool Comparability — HPC Run

**Status:** ⚠️ run complete — race and age flagged (small effects; controls already in place).
**Owner of the run:** whoever has the CSpineSeg data mounted (HPC). Paste the
output into [§ Results](#results-paste-hpc-output-here) below.

---

## Why we are doing this

The whole biased-ruler story — and Aasa's escalation of it into a community-level
data-leakage warning to the CSpineSeg authors — rests on **one assumption**:

> Zhou et al. chose which exams to expert-annotate **pseudo-randomly** (by
> medical-record number, not by demographics or difficulty), so the gold and
> silver pools are demographically comparable, and a gold-vs-silver gap reflects
> **label provenance**, not a difference in case mix.

This single assumption carries:

- **E3 / E4** (the bias-amplification comparisons) — `M_gold` vs `M_silver` train
  on disjoint pools, so any fairness difference is confounded if the pools differ.
- The **dataset-section comparability sentence** (`submission/sections/dataset.tex`,
  currently a `% TODO` that *asserts* comparability without backing it).
- **Aasa's leakage letter** and the headline **~8-Dice-point** inflation number.

A reviewer's first objection is *"maybe the gold cases were just harder / just
different."* Right now we **assert** comparability; this run **shows** it.

### What is already covered (don't redo)

The **anatomical-difficulty** half is already done locally —
`outputs/eda/gold_silver_inspection/morphology/stats.json` compares the pools on
voxel counts, boundary ratio, and connected-component counts (6 of 7 features
n.s.; the one "significant" hit, VB component count, has identical medians 8 vs 8
and would not survive correction). That is reassuring on *difficulty*.

### What this run adds (the gap)

The **demographic + scanner** half is computed **nowhere** — `cohort_composition.json`
is the whole cohort, and `split_v3_{gold,silver}/stats.json` are empty `{}`. This
run fills it: does the gold pool match the silver pool on **sex, race, age,
manufacturer, and field strength**?

Scanner axes matter specifically because the assignment is really a
**patient-number cutoff** (`gold_cutoff ≈ 481` in the morphology inspection — gold
is roughly the first ~481 patients). If enrollment order coincided with a scanner
upgrade, a hidden confound would hide in manufacturer / field strength — so we
test those explicitly, not just demographics.

---

## What to run

From the repo root, on a machine **with the HPC data mounted** (the raw TSVs and
the `split_v3.tsv` splits file are not on the laptop):

```bash
uv run python scripts/gold_silver_comparability.py
```

It prints a readable table and writes `outputs/eda/gold_silver_comparability.json`.

### Scope of the comparison

Provenance (`annotation_quality`) is read from **`split_v3`**, so the comparison
is on the **analysis cohort D** (the ~1,142 exams that actually feed the models),
*not* the full 1,254-exam working set. That is the right scope for de-risking
E3/E4. (Note: the morphology inspection used the full-D₀ pools 489/766, a slightly
wider scope — if a full-cohort gold-ID list is available, the same comparison can
be widened by swapping the provenance source; not required for the headline claim.)

### How to read the output

- Each attribute gets a test (χ² for categorical, Mann–Whitney for continuous
  age), an **effect size** (Cramér's V or rank-biserial `r_rb`), a raw `p`, and a
  **Benjamini–Hochberg-adjusted `p_fdr`** across the whole family.
- **Comparable** = `p_fdr ≥ 0.05` **and** effect size `< 0.1` (negligible band).
- **Desired result:** every row `comparable`, i.e. the pools match on every
  marginal → the pseudo-random-assignment assumption holds → the dataset
  comparability sentence is backed, the reviewer objection is pre-empted.
- **If any row flags `** CHECK **`:** that marginal differs between pools. This is
  important either way — if (say) manufacturer differs, the gold/silver gap is
  partly a scanner effect and the comparability claim must be **softened or
  controlled for**, not asserted. Report it honestly; don't bury it.

---

## Results — paste HPC output here

> **For the LLM/operator running this on the HPC:** run the command above, then
> fill in the three blocks below and flip the status at the top to ✅ (all
> comparable) or ⚠️ (something flagged). Keep the raw table verbatim so the
> numbers are auditable.

### 1. Raw console table

```
Analysis cohort D (split_v3): 1142 exams = 448 gold + 694 silver

Attribute                       test        effect        p    p_fdr  verdict
------------------------------------------------------------------------------------
Sex                             chi2cramers_v=+0.034    0.250    0.291  comparable
Manufacturer                    chi2cramers_v=+0.036    0.228    0.291  comparable
Field strength                  chi2cramers_v=+0.021    0.471    0.471  comparable
Race (White vs. Black)          chi2cramers_v=+0.084    0.007    0.012  ** CHECK **
Race (White/Black/Other)        chi2cramers_v=+0.097    0.004    0.010  ** CHECK **
Age (continuous, years)     mann_whi   r_rb=+0.139    0.000    0.001  ** CHECK **
Age (3-bin)                     chi2cramers_v=+0.122    0.000    0.001  ** CHECK **

  Sex:
     gold    Female 52.2%, Male 47.8%
     silver  Female 48.6%, Male 51.4%
  Manufacturer:
     gold    GE MEDICAL SYSTEMS 35.0%, SIEMENS 65.0%
     silver  GE MEDICAL SYSTEMS 38.8%, SIEMENS 61.2%
  Field strength:
     gold    1.5 58.5%, 3.0 41.5%
     silver  1.5 60.8%, 3.0 39.2%
  Race (White vs. Black):
     gold    Black 24.8%, White 75.2%
     silver  Black 32.9%, White 67.1%
  Race (White/Black/Other):
     gold    Black 22.3%, Other 10.0%, White 67.6%
     silver  Black 30.5%, Other 7.1%, White 62.4%
  Age (3-bin):
     gold    40-60 40.0%, 60+ 35.7%, <40 24.3%
     silver  40-60 37.3%, 60+ 46.4%, <40 16.3%

=> At least one marginal differs -- inspect the flagged row(s).
```

### 2. Verdict

- Overall: **4 / 7 rows flagged** (race and age; sex, manufacturer, field strength comparable)
- Flagged rows and effect sizes:
  - Race (White vs. Black): Cramér's V = 0.084, p\_fdr = 0.012 — gold over-represents White (75.2 % vs 67.1 %)
  - Race (White/Black/Other): Cramér's V = 0.097, p\_fdr = 0.010
  - Age (continuous): |r\_rb| = 0.139, p\_fdr = 0.001 — gold skews younger/middle-aged (60+: 35.7 % vs 46.4 %)
  - Age (3-bin): Cramér's V = 0.122, p\_fdr = 0.001
- `n_gold` = 448, `n_silver` = 694 (analysis cohort D, split\_v3)
- All flagged effect sizes are small (V < 0.13, |r\_rb| < 0.14) but cross the negligible threshold of 0.1.
  The differences are consistent with a patient-number enrollment cutoff coinciding with a demographic
  shift over time, not a deliberate selection.

### 3. Note for `dataset.tex`

**Do not assert blanket comparability.** The current text (lines 120–124) over-claims.
Replace with the softened version already applied (see `submission/sections/dataset.tex`, same lines):
scanner axes are comparable; race and age differ modestly and are already controlled for in all
stratified comparisons.

*(Full comparability sentence dropped because pools differ on race and age. Do not add the
all-clear one-liner. Raise at next supervisor sync — affects E3/E4 framing if not already
controlled.)*

---

## After the run

- [ ] Paste output above; set status ✅/⚠️.
- [ ] Drop the backed sentence into `submission/sections/dataset.tex` (removes the
      `% TODO` at line ~127) and, if relevant, the matching note in
      `paper/sections/methodology.tex` (`sec:regimes`, the pseudo-random claim).
- [ ] If a marginal is flagged, raise it at the next supervisor sync before
      writing the comparability claim — it changes the E3/E4 framing.
