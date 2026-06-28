# Content Cut from `paper/` for the FAIMI Submission

The `submission/` paper is an 8-page FAIMI version compressed from the longer
`paper/` draft. This logs what was **dropped or shortened** per section, so
nothing is lost by accident and anything cut can be reinstated if a reviewer asks
or if a longer venue (e.g. the MELBA special issue) is targeted later.

Source of truth for the full text is always `paper/sections/*.tex` (unchanged).

---

## Dataset and Exploratory Analysis (`dataset.tex`)

**Target:** ~1.25 pp (was ~2.5–3 pp in `paper/`). **Kept:** `fig:anatomy`,
`tab:cohort`, and the gold/silver comparability point.

| Cut from `paper/` | What it was | Why dropped / where it went |
|---|---|---|
| **`tab:notation`** (full notation table) | Big table defining $\mathcal{D}, X_i, Y_i, A_i, q_i, \mathcal{D}_{A=a}, \mathcal{D}^{\mathsf{g}}/\mathcal{D}^{\mathsf{s}}$, splits | Symbols **inlined** in two sentences; gold/silver sets + splits deferred to Methodology where they're first used. Saves a tall float. |
| **`fig:volumes`** (`volume_confounders.png`) | Boxplots of VB/disc volume by age bin | Dropped; the finding is now one prose sentence. **Figure file still exists** for reuse. |
| **`tab:volumes`** (structure-size table) | Gap % + effect band for sex/race/age × VB/disc | **Folded into one sentence**: sex large (both); race/age enlarge VB only (small/medium); disc unchanged. |
| **Connected-components paragraph** | VB label splits into fewer pieces with age (median 9→7, medium effect); disc flat — the "degeneration fingerprint" | **Fully cut.** Note: this is independent support for the *intrinsic age-difficulty* argument — worth reinstating in Discussion if we need to reinforce that the age effect is anatomy, not label noise. |
| **GE/Siemens voxel detail** | Per-label voxel medians (69,689 vs 32,401), VB mm³ medians (49,850 vs 52,956), disc n.s. | **Compressed to one line** keeping only the ~2.15× voxel ratio and the ~6% mm³ gap. |
| **Effect-size footnote** | Caveat that Cohen's bands derive from Pearson $r$, applied to $r_{rb}$ by convention | Dropped (convention is standard; saves a footnote). |
| **Parikh breast-MRI analogue sentence** + some framing prose | "younger patients' larger tumours make those cases intrinsically harder…" and the "confounder, not a finding" line | Trimmed for length; the analogy lives in Intro/Discussion instead. |

**Added (not in `paper/` dataset section):** the **gold/silver demographic
comparability** statement was surfaced here from `paper/` methodology
(`sec:regimes`), because it is load-bearing for the E2 leakage claim. Still
carries a `% TODO` to back it with an empirical marginal-match line.

---

## Methodology (`methodology.tex`)

**Target:** ~1.5 pp (was ~520 lines in `paper/`). **Structure:** 6 subsections →
2 subsections + `\paragraph` run-ins. **Kept:** `fig:regimes` (author's partition
figure), `tab:experiments`, and the nDSC equation.

| Cut / shortened from `paper/` | What it was | Why dropped / where it went |
|---|---|---|
| **`tab:regimes`** (regime composition table) | Train/val/test counts per regime | Dropped in favour of the author's **partition figure** (`fig:regimes`), which shows the same gold⊕silver→regime split; train counts (288/450/798) kept in prose, test counts in `tab:experiments`. (`paper/`'s TikZ `fig:regimes` is also superseded by this figure.) |
| **Split-portability prose** | Paragraph on why keep 70/10/20 + the validation tier for non-CV-native models | Cut to one clause ("nnU-Net is CV-native, so we pool train+val"). |
| **Sex-balance arithmetic** | Exact cancellation (gold 174F/144M, silver 255M/225F, the 30-surplus per pool) | Compressed to one sentence: counts don't sum because each regime balances independently. |
| **Gold/silver imbalance paragraph** | Full argument that the pools are demographically comparable (Zhou's MRN assignment) | Compressed to one clause; the substance now lives in `sec:confounders` (dataset). |
| **Equations dice/load/rate/dpddir** | Separate display eqs for Dice, reference load $r_c$, success rate $\rho_a$, DPD/DIR | **Inlined**; only `eq:ndsc` kept as a display equation. |
| **"Which instrument is primary" paragraph** | Standalone argument that continuous tests beat DIR in a high-overlap regime | Compressed to the last sentence of the metrics paragraph. |
| **Post-proc / normalisation detail** | Per-case z-score normalisation; the VB Dice $0.96\to0.49$ figure for per-class cleanup | Dropped; kept only "global largest-foreground cleanup". |
| **Generated-silver-ruler subsection** | Long E2 ruler-construction prose | Folded into the `\paragraph{Experimental design}` beat. |

**Note:** `sec:metrics` and `sec:design` labels now sit on their `\paragraph`s (not
the enclosing subsection), so `\nameref`s from `dataset.tex` resolve to the
paragraph titles.

## Experiments (`experiments.tex`)

**Done.** E1 compressed to a short setup; E2 given the most room (headline);
E3–E4 folded into prose; probe removed (see scope cut below). **Final floats:**
1 figure (`age_trend`) + 1 table (`biased_ruler`).

| Cut from `paper/` | What it was | Why dropped / where it went |
|---|---|---|
| **`fig:global_audit`** (`dir_forest_global`) | E1 forest plot | First folded into `tab:global_dir`; then `tab:global_dir` itself cut in the 8pp pass — E1 is now fully prose. File still exists. |
| **`violin_dice_macro_age`** | E1 age-distribution violin | Dropped (age distribution lives in the E2 figure). File still exists. |
| **`tab:amplification`** (E3–E4 DIR table) | 3 models × 5 groupings macro-Dice DIR | **Folded into one `\paragraph`** of prose (M_silver≡M_mix, M_gold 0.888, all DIR>0.93, no amplification). |
| **`probe_bar_chart`, `tab:probe`** | Probe AUROC chart + table | Dropped — the entire probe thread was removed (see scope cut below). |

## Page-budget cuts — first pass (2026-06-27)

The three written sections overran ~3pp. Cuts made, no meaning lost:

| Change | Section | Saving |
|---|---|---|
| Cut `fig:dir_forest` (E1 forest) — redundant with `tab:global_dir` | experiments | ~0.35pp |
| Shrunk `fig:anatomy` 0.55→0.4 textwidth (**later dropped entirely**, see below) | dataset | ~0.2pp |
| Dropped `tab:probe` Manufacturer rows → prose clause | experiments | minor |
| Compressed cohort prose (percentages now only in `tab:cohort`) | dataset | ~0.15pp |
| Tightened `sec:confounders` intro + scanner/age + "two choices" paras | dataset | ~0.25pp |
| Dropped count-doesn't-sum explanation from regimes prose (now in `fig:regimes` caption) | methodology | ~0.15pp |

## Page-budget cuts — aggressive 8pp pass (2026-06-28)

Still ~3pp over after the first pass (the unwritten abstract/intro/conclusion add
~1.5pp). Decision: **cut hard, stay at FAIMI 8pp** (not MELBA). Floats went 9 → 5
(2 fig + 3 tbl). Body text 4,147w → 3,245w. Cuts, by section:

| Change | Section | Saving |
|---|---|---|
| **Dropped `eq:ndsc`** (the nDSC κ-reweighting display equation + κ definition) → nDSC now one prose clause, citing Raina et al. ⚠️ flagged for possible reinstatement | methodology | ~0.15pp |
| **Dropped `tab:global_dir`** (E1 DIR/DPD×metrics table) → E1 "all fair" verdict folded to one prose sentence (all DIR>0.96, all BCa lower bounds >0.80, 0/63 sig) | experiments | ~0.3pp |
| **Dropped `fig:anatomy`** entirely (the shrink in the first pass wasn't enough) | dataset | ~0.45pp total |
| **Removed the entire probe thread** (own section below) | all | ~0.5–0.6pp |
| Trimmed "Model & training" para (cut the "serves the audit twice over" justification; GPU/CO₂ → one clause) | methodology | ~0.2pp |
| Compressed the metrics/inference battery (DPD/DIR/MWU/KW/BCa/OLS/permutation) to its essentials | methodology | ~0.2pp |
| Compressed the E1–E4 experimental-design para (E2 ruler-construction detail shortened) | methodology | ~0.15pp |
| Second-pass compression of `sec:confounders` (volume analysis + gold/silver comparability prose) | dataset | ~0.25pp |

**Still kept:** `fig:regimes` (the partition diagram — author values its clarity),
`fig:age_trend`, `tab:cohort`, `tab:experiments`, `tab:biased_ruler`.

**Still open (candidate further cuts if needed):** reinstate `eq:ndsc` only if space
allows. (`tab:experiments` and the `tab:cohort` shrink were actioned in the
2-page pass below.)

## Scope cut: encoder probing removed entirely (2026-06-28)

The whole probing thread was cut to sharpen the paper to a single thesis (the
biased ruler / false confidence) and reclaim ~0.5–0.6pp. It was the most off-axis
content — representation analysis in an outputs-and-rulers paper. Removed across
**every** section so nothing dangles:

| Location | What was removed |
|---|---|
| `methodology.tex` | the `\paragraph{Encoder probing}` (+ `\label{sec:probing_method}`) |
| `experiments.tex` | the `\paragraph{Demographic probing}`, `tab:probe`, and the intro-line mention |
| `discussion.tex` | retitled para 2 to "The age effect is intrinsic, not unfairness" (kept the age-intrinsic argument, which stands on both-rulers + anatomy); dropped the encoding-dissociation sentences and **recommendation (4)** ("don't strip demographic signal…") — now three recommendations |
| `introduction.tex` (beats) | Gichoya/Petersen related-work line; the "encoding ≠ disparity" half of contribution 4 |
| `abstract.tex` / `conclusion.tex` (beats) | probe/encoding clauses |

**Now-unused citations** (harmless — splncs04 prints only cited entries; left in
`mybibliography.bib`): `petersen2023feature`, `gichoya2022reading`, `dong2025mricore`.

**Reinstating** (for a longer venue, e.g. MELBA): the full probe lives in
`paper/sections/experiments.tex` (`tab:probe`, both encoders, nulls) and
`paper/sections/discussion.tex` (`sec:encoding_disparity`). The figure
`probe_bar_chart.png` still exists.

**Note:** `eq:ndsc` was also dropped in the page-budget pass above (nDSC now
described in one prose clause) — flagged for possible reinstatement.

## Discussion (`discussion.tex`)

**Done.** `paper/`'s 7 subsections → 5 `\paragraph` run-ins; limitations and
recommendations are **compact inline lists** (cheaper than `enumerate`); future
work trimmed to one closing sentence on the recommendations paragraph.

| Cut / shortened from `paper/` | Why dropped / where it went |
|---|---|
| Per-mode `\subsection`s (ruler modes, age-intrinsic, no-amplification, encoding, limits, future, recs) | Collapsed to `\paragraph`s to drop subsection whitespace. |
| Standalone **Future Work** subsection (TotalSpineSeg, extended-label re-audit) | Cut to one clause (independent/deliberately-biased ruler + variance-collapse diagnostic) on the recommendations paragraph. |
| Long encoding/Gichoya discussion | **Removed entirely** with the probe scope cut; the age-intrinsic argument it supported now stands on both-rulers agreement + the anatomy confounder. |
| Recommendations as a 4-item `enumerate` with full prose | Inlined as bolded run-ins; now **three** — recommendation (4) ("don't strip demographic signal…") cut with the probe. |

**Adapted (not a straight cut):** the "gold/silver pools are demographically
comparable" claim was refined after the empirical check — the pools match on sex
and scanner but **differ modestly on race/age** (gold Whiter/younger). Discussion
now carries this as a limitation (E3/E4 confounds provenance with composition; the
E2 ruler result is unaffected, same images). Mirrors the edits in `dataset.tex`
(`sec:confounders`) and `methodology.tex` (`sec:regimes`).

**Reframed leakage-forward (2026-06-28, follow Aasa's letter):** the "Two modes of
ruler bias" paragraph now states **label leakage** as the cause of the ~8-pt offset
and the variance collapse (both models trained partly on the same gold labels → they
resemble each other more than truth); the old footnote attributing the agreement to
"task simplicity / not training overlap" was **removed** (its inference was wrong —
see [`../fairness/biased-ruler-silver-source.md`](../fairness/biased-ruler-silver-source.md))
and folded into the main text as evidence *for* leakage. Recommendations gained a new
leading item, **(1) report against expert labels** (Aasa's fix), so there are now four.
Mirrored in `experiments.tex` (E2 retitled, mechanism named).

## Two-page pass to hit 8pp (2026-06-28, round 2)

Draft was ~10 content pages after the abstract/intro/conclusion were written. Target
~2pp. Cuts made, no findings lost:

| Change | Section | Est. saving |
|---|---|---|
| **Dropped `tab:experiments`** — every cell (question/model/test set/ruler for E1–E4) folded into the `\paragraph{Experimental design}` prose with bold E1–E4 run-ins; `\autoref{tab:experiments}` in `experiments.tex` → `\nameref{sec:design}` | methodology | ~0.3pp |
| **Dropped `fig:regimes`** (the TikZ partition diagram) — its content (gold/silver→3 models, counts $798$=$318$g+$480$s, $288$g, $450$s, shared recipe/split) inlined into the regimes prose. *User approved cutting it after initially keeping it.* (`\usepackage{tikz}` + `\tikzset` left in `main.tex`, now unused but harmless.) | methodology | ~0.5pp |
| **Compressed `sec:confounders`** 4 paras → 3 (folded "two choices" into the volume para) **then slashed to a single paragraph** (dropped the per-group volume %s and the effect-size cite cluster; kept disc cross-check, nDSC rationale, age-for-race control, gold/silver E3/E4 caveat) | dataset | ~0.5pp |
| **Contributions `enumerate` → run-in prose** ((1)–(5) bolded, same content, no list whitespace) | introduction | ~0.35pp |
| **Shrank `tab:cohort`** — removed the Ethnicity rows and both footnotes (→ one caption sentence); rebalanced Field-strength to the left column | dataset | ~0.2pp |
| Trimmed `\paragraph{Model and training}` (justification clauses) and the metrics/stat-battery para (BCa→bootstrap, nDSC wording) | methodology | ~0.25pp |
| Discussion `\paragraph{Limitations}` 6 items → 4 (merged single-institution/anatomy/architecture/test-set) | discussion | ~0.1pp |

Body text 3,996w → 3,722w on top of the float removals. **Protected (not touched):**
`fig:age_trend`, `tab:biased_ruler`, abstract, conclusion, the full E1–E4 thread, and the
leakage prose — the audit + E2 / Aasa-story payload.
**Next lever if still over:** drop `tab:cohort` entirely to prose (~0.3pp), or cut the
E3/E4 amplification thread (~0.6pp, removes contribution #4). Re-measure on the next build
(no local LaTeX here).

## Scope cut: E3/E4 amplification thread removed (2026-06-28, round 3)

Still ~0.75pp over after round 2. Decision criterion: **cut the parts that support
Aasa's leakage thesis the least.** The clear answer was the **E3/E4 bias-amplification
experiment** — it answers a *different* question (does *training* on silver amplify
bias) and is a null; it was Aditya's boundary-condition contribution, not Aasa's
leakage story. Removed across every section (~390 body words, ~0.7pp):

| Location | What was removed |
|---|---|
| `experiments.tex` | the whole `\paragraph{Bias amplification (E3--E4)}`; "E1--E4" → "E1 and E2" |
| `methodology.tex` | `\paragraph{Experimental design}` retitled E1--E4 → **E1--E2**, E3/E4 sentences dropped; regimes para reframed — `M_silver` recast from amplification subject to **leakage control** (the `M_gold`/`M_silver` disjoint-image agreement is kept as evidence that leakage is label-level) |
| `introduction.tex` | contribution **#4 (boundary condition on amplification)** dropped; now 4 contributions |
| `discussion.tex` | the whole `\paragraph{Why silver training did not amplify bias}`; limitation **(iv)** (the E3/E4 composition/size confound) dropped → 3 limitations |
| `dataset.tex` (`sec:confounders`) | the gold/silver pool-comparability sentence (existed only to de-risk E3/E4; E2 ruler is immune regardless) |
| `abstract.tex`, `conclusion.tex` | the "training on silver does not amplify bias" clauses |

**Kept (deliberately):** `M_silver` survives as a trained model because the
`M_gold`-vs-`M_silver` agreement (~0.97 on disjoint images) is one of the *strongest*
arguments **for** Aasa's thesis — it shows the leakage is in the labels, not shared
training images. The intro still notes Parikh et al.'s amplification finding as *lineage*
(we extend their ruler effect, not their amplification result).

**Reinstating:** the full amplification experiment (Run 9, no amplification) lives in
`paper/sections/{experiments,discussion}.tex` and `docs/.../06_gold_silver_training.md`.
For a longer venue (MELBA), restore contribution #4 + both paragraphs.

**Heads-up for Aditya:** this removes the boundary-condition result he valued. It is
reversible; flag at the next sync.

**Final reserve lever** if the build is still a sliver over: `tab:cohort` → 2 prose
sentences (~0.3pp).

## Abstract / Introduction / Conclusion (2026-06-28)

**No longer skeletons — drafted leakage-forward.** `abstract.tex`, `introduction.tex`
(with the 5-item contributions list filled in), and `conclusion.tex` are written.
Ordering follows Aasa's story: contribution #2 is the leakage/overestimation (~8 Dice
points, affects any dataset user), #3 is the false-confidence verdict-flip (the
fairness-specific consequence of the same leakage), then the amplification boundary
condition and the practitioner guidance. **Page-budget note:** the earlier estimate
that "the unwritten abstract/intro/conclusion add ~1.5pp" is now realised — re-measure
the 8pp budget on the next build (no local LaTeX toolchain to check here); the
"candidate further cuts" listed above (drop `tab:experiments`, shrink `tab:cohort`,
keep `eq:ndsc` out) are the first levers if it overflows.
