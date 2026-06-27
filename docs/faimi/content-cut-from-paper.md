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

## Experiments (`experiments.tex`) — TBD

_Planned: E1 compressed to a short setup; E3–E4 + probe demoted to `\paragraph`s;
drop the standalone `violin_dice_macro_age` (fold into the E2 figure) and
`probe_bar_chart` (table suffices)._

## Discussion (`discussion.tex`) — TBD

_Planned: subsections → `\paragraph` run-ins; limitations/recommendations kept as
compact lists; future-work trimmed to 1–2 sentences if space allows._
