# Biased Ruler v2 — Independent Silver Ruler (Aditya's proposal, 2026-06-12)

## Motivation

The current biased ruler (E2) evaluates $M_\text{mix}$ against two rulers:
- Ruler A: gold expert labels
- Ruler B: $M_\text{gold}$'s predictions (generated silver)

**Problem:** $M_\text{mix}$ and $M_\text{gold}$ share 318 gold training cases →
they are "correlated twins" (agreement ~0.973) → the silver ruler saturates
(DIR ≡ 1.0 at τ=0.8, zero failures). The biased-ruler signal only shows up in
continuous tests (variance collapse → false confidence), not in the binarized DIR.

## Aditya's Proposal

Audit **$M_\text{gold}$** instead, using **$M_\text{silver}$'s predictions** as
the silver ruler:

| | Model under audit | Ruler A (gold) | Ruler B (silver) |
|---|---|---|---|
| **Current (E2)** | $M_\text{mix}$ | expert labels | $M_\text{gold}$ predictions |
| **Proposed (v2)** | $M_\text{gold}$ | expert labels | $M_\text{silver}$ predictions |

**Why this may be better:**
- $M_\text{gold}$ and $M_\text{silver}$ train on **completely disjoint** data
  (288 gold vs 450 silver cases, zero overlap)
- No shared training data → no twin effect → agreement should be lower (~0.90–0.95
  instead of 0.97) → more variance → DIR may not saturate
- Closer to Parikh et al.'s setup where the silver annotator is an independent model

## What We Need to Check First

**Is it still degenerate?** Even with disjoint training, both models solve the
same easy task (Dice ~0.89 against truth). If inter-model agreement is still >0.95,
the binarized DIR will still saturate.

**Quick check:** Compute $M_\text{gold}$ predictions vs $M_\text{silver}$
predictions on the 76 gold-test images. If macro Dice is:
- <0.92: non-degenerate, proceed with full fairness analysis
- 0.92–0.95: borderline, check how many cases fall below 0.8
- >0.95: likely still saturates, same problem as current E2

## Implementation

### Step 1: Evaluate agreement (diagnostic)

```bash
# Score M_gold predictions against M_silver predictions (as reference)
uv run python -u -m src.fairness.evaluate \
    --predictions "${nnUNet_results}/Dataset002_CSpineSeg_Gold/predictions_test_pp" \
    --references  "${nnUNet_results}/Dataset003_CSpineSeg_Silver/predictions_gold_test_pp" \
    --mapping     "${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json" \
    --output      outputs/eval_ds2_vs_ds3.csv \
    --metrics dice hd95 ndsc \
    --workers 24
```

If results look promising (meaningful failures at τ=0.8), proceed to Step 2.

### Step 2: Full biased ruler v2 analysis

```bash
# M_gold vs gold labels (already exists: outputs/eval_ds2_on_gold.csv)
# M_gold vs M_silver predictions (new: outputs/eval_ds2_vs_ds3.csv)

uv run python -u -m src.fairness.analyze \
    --evaluation-csvs outputs/eval_ds2_on_gold.csv outputs/eval_ds2_vs_ds3.csv \
    --ruler-labels    gold silver \
    --mapping         "${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json" \
    --report-name     fairness_biased_ruler_v2
```

## Expected Outcomes

1. **Non-degenerate (agreement ~0.90–0.93):** We get a proper biased-ruler
   comparison where the binarized DIR actually moves. Could show magnitude
   inflation (Parikh's mechanism) rather than just false confidence.

2. **Still degenerate (agreement >0.95):** Confirms the fundamental issue is
   task simplicity — cervical spine segmentation is too easy for *any* silver
   ruler to produce meaningful binarized-DIR movement. The contribution stays
   as the false-confidence mechanism (current E2).

## Decision

- [ ] Run Step 1 diagnostic on HPC
- [ ] Review agreement numbers
- [ ] Decide whether to run full analysis or keep current E2
