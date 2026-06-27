# Gold vs. Silver Pool Comparability — HPC Run

**Status:** ⏳ script written, awaiting HPC run.
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
<paste the full printed table + the per-pool percentage breakdown here>
```

### 2. Verdict

- Overall: `<all comparable / N rows flagged>`
- Flagged rows (if any) and their effect sizes: `<...>`
- `n_gold` / `n_silver` in the analysis cohort: `<...>`

### 3. One-liner for `dataset.tex`

> Replace the `% TODO` at `submission/sections/dataset.tex` (gold/silver
> comparability) with a sentence backed by these numbers. Draft to adapt once the
> numbers are in:

```
The two pools are demographically and acquisition-matched: gold and silver differ
on none of sex, race, age, manufacturer, or field strength (all p_fdr > 0.05, all
Cramér's V / |r_rb| < 0.1), confirming that label provenance — not case mix — is
the only systematic difference between them.
```

*(Adjust wording if any marginal is flagged: state which one, its effect size,
and how it is controlled for.)*

---

## After the run

- [ ] Paste output above; set status ✅/⚠️.
- [ ] Drop the backed sentence into `submission/sections/dataset.tex` (removes the
      `% TODO` at line ~127) and, if relevant, the matching note in
      `paper/sections/methodology.tex` (`sec:regimes`, the pseudo-random claim).
- [ ] If a marginal is flagged, raise it at the next supervisor sync before
      writing the comparability claim — it changes the E3/E4 framing.
