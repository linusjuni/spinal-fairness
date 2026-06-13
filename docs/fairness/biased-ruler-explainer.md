# Biased Ruler — Intuitive Explainer

> A simple explanation of the two biased-ruler modes, without jargon.

## Setup

- **Student** = your segmentation model (the thing being judged)
- **Template** = reference label (the thing you grade against — the "ruler")
- **The drawing** = a segmentation of a specific patient's image
- **Gold template** = expert-drawn ground truth (perfect ruler)
- **Silver template** = another model's predictions used as reference (potentially biased ruler)

The fairness question is: "does the student's drawing score differently for different patient groups (e.g. young vs old)?"

---

## Mode 1: False Magnitude (Parikh et al., breast MRI)

**The silver ruler is an independent model that is itself worse at one subgroup.**

### What happens

1. Grade student against **gold** → young patients score 87, old patients score 90. Gap = 3 points. DIR = 0.871. "Hmm, borderline."
2. Grade student against **silver** → young patients score 82, old patients score 90. Gap = 8 points. DIR = 0.815. "Definitely unfair!"

### Why

The silver template is a different model that is *itself* bad at young patients. So on a young patient:

- The student drew the segmentation slightly wrong (shifted left)
- The silver template *also* drew it wrong, but in a *different direction* (shifted right)
- Dice measures overlap → left + right shift = big disagreement

On an old patient:

- The student drew it almost right
- The silver template also drew it almost right
- They agree fine → high Dice

The errors don't cancel — they compound. It's like two people both misremembering a phone number but in different digits: they disagree with *each other* more than either disagrees with the actual number.

### Why it's a problem

The model didn't change. The patients didn't change. You just swapped the ruler. Now you'd publish "this model discriminates against young patients" with an effect size that's **double** the reality. A policymaker would demand fixes to a problem you exaggerated. Or the reverse: a lenient ruler could *hide* a real disparity.

### Summary

The silver ruler **inflates the signal** (makes the gap look bigger than it is).

---

## Mode 2: False Confidence (our work, cervical spine)

**The silver ruler is a near-clone of the model being tested — they agree on almost everything.**

### What happens

1. Grade student against **gold** → all age groups score ~90, but scores jump around a lot (std = 5.8 pts). Between-group difference = 0.7 pts. Statistical test says "not significant." Verdict: **fair.**
2. Grade student against **silver** → all age groups score ~97, scores barely move (std = 1.4 pts). Between-group difference = still 0.7 pts. Statistical test says "significant! p < 0.05!" Verdict: **unfair!**

### Why

The silver template is a near-twin of the student (same architecture, same training recipe, same task). They make the same mistakes on the same patients. So:

- Against gold: scores are [0.82, 0.91, 0.88, 0.95, 0.79, 0.93, ...] — lots of spread
- Against silver: scores are [0.97, 0.98, 0.96, 0.98, 0.97, 0.97, ...] — tight cluster

The statistical test asks: *"is the between-group difference large relative to the within-group spread?"*

- Gold: signal (0.7 pts) / noise (5.8 pts) ≈ 0.12 → "can't tell" → not significant
- Silver: signal (0.7 pts) / noise (1.4 pts) ≈ 0.50 → "clearly different" → significant

Same signal. Different noise floor. The silver ruler didn't find a problem — it removed the noise that was (correctly) preventing you from calling a trivial 0.7% gap "significant."

### Why it's a problem

You'd publish "we found a statistically significant age disparity" when in reality the gap is clinically meaningless (every patient gets >97% Dice). The ruler manufactured certainty from nothing.

### Summary

The silver ruler **removes noise** (makes a negligible gap look statistically detectable).

---

## Side-by-side

| | Mode 1 (Parikh) | Mode 2 (ours) |
|---|---|---|
| Silver template is... | independently biased (bad at same subgroup) | a near-clone (agrees on everything) |
| Effect on the gap | makes it **bigger** | keeps it the same or smaller |
| Effect on the noise | unchanged | **collapses** it |
| Why verdict flips | signal grows → crosses threshold | noise shrinks → same signal crosses threshold |
| Danger | exaggerates a real problem | manufactures significance from a non-problem |

---

## What if the model is fair but the ruler is biased?

This is the purest case to build intuition from. Your model segments equally well for all groups — it's perfectly fair. But your silver labels are biased (e.g. worse quality for older patients).

You compute Dice(model prediction, silver label) per patient. Since Dice measures *agreement between two things*:

- For young patients: model is correct, silver is also correct → high agreement → high Dice
- For old patients: model is correct, but silver is wrong → low agreement → low Dice

**Incorrect conclusion:** "The model performs worse on older patients! Unfair!"

But the model was fine — it's the *ruler* that's worse for older patients. You're blaming the painter for a crooked measuring tape.

### The two failure modes (fair model + biased ruler)

| Ruler bias direction vs model | What you see | Reality |
|---|---|---|
| Ruler biased, model fair | "Model is unfair!" | False alarm — ruler's bias projected onto model |
| Ruler biased same way as model | "Model is fair!" | Missed finding — ruler can't see the shared blind spot |

The second row is equally dangerous: if the silver labels are biased in the *same direction* as the model (both worse on old patients, by the same amount), they'd *agree* on old patients (both wrong in the same way → high Dice everywhere). Verdict: "fair!" — but both the model and the ruler are failing older patients together, hiding each other's bias.

**This is why gold labels matter:** they're the only reference that tells you about the *model* rather than about the *agreement between two imperfect systems*.

---

## Our case: both fair, but variance collapses

Our case is neither of the rows above. Gold confirms the model is fair. The silver model is also fair (no demographic bias). So what goes wrong?

The problem isn't bias in the ruler — it's **correlation**. Both models are fair, both are good, and they make the same tiny errors on the same patients (same easy task, same architecture). That extreme agreement collapses the variance. A real but clinically meaningless age trend (old spines are slightly harder — more degeneration) then sticks out because the noise floor is gone.

- Against gold: signal (0.7 pts) / noise (5.8 pts) → can't tell → not significant → **correct verdict: fair**
- Against silver: signal (0.7 pts) / noise (1.4 pts) → stands out → significant → **incorrect verdict: unfair**

The gold ruler's noise was doing you a *favour*: it correctly reflected that the gap is too small to care about given the overall measurement uncertainty. Like using a microgram scale to weigh two people and declaring them "significantly different" because one weighs 0.001g more.

---

## How the statistical testing works (our setup)

For each demographic attribute, we ask: "is there a performance difference between groups?"

- **3+ groups** (e.g. age: <40, 40–60, 60+) → **Kruskal-Wallis** test (omnibus: "is at least one group different from the rest?")
- **2 groups** (e.g. age median split, sex) → **Mann-Whitney U** test ("are these two groups different?")

We repeat this for every combination of:
- **7 demographic groupings** (sex, race_wb, race_wbo, race_wn, ethnicity, age_3bin, age_median)
- **9 metrics** (Dice / nDSC / HD95 × VB / disc / macro)

= **63 tests total**. We apply Benjamini-Hochberg FDR correction across all 63 to control for multiple comparisons.

### Results

- **Gold ruler: 0/63 significant.** Nothing. Fair across all attributes and metrics.
- **Silver ruler: 11/63 significant.** Every single one is an age comparison:

| Metric | Structure | Age grouping | $p_{\text{fdr}}$ |
|--------|-----------|-------------|---------|
| Dice | VB | 3-bin | 0.033 |
| Dice | Disc | 3-bin | 0.032 |
| Dice | Macro | 3-bin | 0.026 |
| nDSC | VB | 3-bin | 0.009 |
| nDSC | Disc | 3-bin | 0.014 |
| nDSC | Macro | 3-bin | 0.009 |
| Dice | Disc | median split | 0.024 |
| Dice | Macro | median split | 0.037 |
| nDSC | VB | median split | 0.024 |
| nDSC | Disc | median split | 0.012 |
| nDSC | Macro | median split | 0.012 |

**Pattern:** only age (sex and race never significant on either ruler), only overlap metrics (Dice/nDSC — HD95's heavy-tailed variance is too large even on silver), both structures, both age groupings.

---

## The one-sentence takeaway

> A fairness audit grades a model's predictions against reference labels — but if those labels carry their own demographic blind spots (Mode 1) or are too similar to the model being tested (Mode 2), the verdict tells you about the ruler as much as about the model.
