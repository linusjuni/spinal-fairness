# Archived fairness analyses (old mean-based DPD/DIR)

These runs were computed under the **old continuous, mean-based** DPD/DIR
definition, superseded on 2026-06-07 by the binarized, rate-based definition
(four-fifths / Parikh et al.). See `docs/fairness/dpd-dir-redefinition.md`.

Kept for reference only — **do not cite these numbers**. Per-case metric CSVs
are unchanged by the redefinition; only the aggregation differs.

| Dir | What | Superseded by |
|---|---|---|
| `fairness/` | Global audit Runs 1–5 (Dice / +HD95 / +nDSC) | `outputs/fairness/fairness_global/` (new def) |
| `biased_ruler/` | Biased-ruler Run 6, June 5 — **only complete gold+silver+widening result so far** | `outputs/fairness/fairness_biased_ruler/` once its silver-ruler crash is fixed and rerun |

Note: `biased_ruler/20260605_121654` is still the only end-to-end biased-ruler
result (both rulers + `dir_widening_*`). The June-7 new-definition rerun
(`../fairness_biased_ruler/`) crashed on the silver ruler and produced gold only.
