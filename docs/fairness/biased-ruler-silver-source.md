# Biased Ruler — Silver-Source Limitation & Options (decision note, 2026-06-09)

> **Update (2026-06-10) — Option 2 executed.** The bias-amplification experiment
> (Dataset002 gold vs Dataset003 silver, Run 9) is now done and found **no amplification**:
> silver-trained DIRs match mixed-trained across all groupings, 0/63 FDR on all three models
> (`fairness-runs.md` Run 9, `06_gold_silver_training.md`). This resolves the "is the gap
> amplified by *training* on silver" half of the question — it is not, unlike Parikh et al.
> Exp 4 (our silver is a correlated, high-quality twin, so there is little demographic bias to
> amplify). The **Options/Recommendation** below are kept as the historical decision record;
> the only still-open item is whether to also demonstrate *magnitude inflation* with a
> deliberately biased independent silver (Options 3/4) — a supervisor-meeting call.

## Context

Our biased-ruler experiment (Run 8, [`fairness-runs.md`](fairness-runs.md)) takes the
mixed model (Dataset001), runs it once on the 76 gold test images, and scores those
predictions against **two rulers**: gold expert labels, and "generated silver" labels =
**Dataset002's predictions** on the same images. The intent (after Parikh et al.) is to
show that the choice of reference label can distort a fairness verdict.

It does — we **reproduce Parikh et al.'s central claim** (the choice of reference label
flips the fairness verdict) but through a **complementary mechanism**: our correlated
silver ruler manufactures *false confidence* rather than *false magnitude*. This note
records the mechanism, how our result compares to the source papers, the genuine limits
of our setup, and the options going forward. **The decision is open and is a
supervisor-meeting item.**

> **Read this first — the model is fair.** Nothing below says the model disadvantages a
> demographic group. By every practical measure the audit is **fair across sex, race, and
> age** (sub-percent gaps, DIR ≫ 0.8). The "significant age effect against silver" is a
> *methodological* result about how a flawed ruler can manufacture significance from a
> clinically irrelevant gap — **not** a finding that older patients are failed. Significance
> ≠ unfairness; see the dedicated section below before quoting any "age" number.

## What we actually find vs Parikh et al.

| | Parikh et al. (MAMA-MIA, breast) | This work (CSpineSeg, cervical) |
|---|---|---|
| Salient axis | **Age** | **Age** (same) |
| Direction | **Young** worst | **Old (60+)** worst — *flipped* |
| Magnitude (gold/true) | large, robust; raw-cohort **DIR → 0.699** | tiny, clinically negligible; all Dice **DIR ≥ 0.94** |
| **Verdict flip** (gold → silver ruler) | significant → **more** significant (gap +40%, DIR 0.871 → 0.815) | **not significant (0/63) → significant (11/63, all age)** |
| Biased-ruler mechanism | **false magnitude** — silver inflates the gap's *size* | **false confidence** — silver inflates the gap's *certainty* |
| Amplification (train on silver) | confirmed: gap +66%, **DIR → 0.79** | **no amplification** — silver-trained DIRs ≈ mixed, 0/63 FDR (Run 9, 2026-06-10) |

**We reproduce the core biased-ruler effect.** Parikh's thesis is "a flawed ruler
misrepresents true bias." On our 76 images, the *same model* is judged **age-fair against
gold (0/63 FDR)** but **age-unfair against silver (11/63, all age)** — the significance
verdict reverses purely from the reference label. That *is* a biased-ruler effect. It just
runs through a different mechanism than Parikh's, and the direction flip (young↔old) is
physiologically coherent (breast: young = denser/harder; cervical spine: old = more
degenerative/harder).

## Root cause: silver provenance → false magnitude vs false confidence

The biased-ruler effect's *mechanism* depends on where the silver labels come from:

- **Parikh et al.'s silver is an independent, genuinely biased annotator** (a model
  trained on *external* data, itself worse for young patients). Grading against it
  *compounds* the deficit where the patient is hard → it inflates the gap's **magnitude**
  (DIR 0.871 → 0.815).
- **Our silver is a near-twin of the model under test.** Dataset002 agrees with
  Dataset001 ≈**0.97**, while *each* model agrees with the gold truth only ≈**0.89**
  (DS001-vs-gold 0.897, DS002-vs-gold 0.888) — the two share correlated systematic
  deviations, so "near-twin" is empirically supported, not assumed. The consequence is
  **two-sided**, and "masking" describes only one side:
  - *On the DIR lens:* agreement is so high and tight that every case clears 0.8 → DIR
    saturates to **1.0** → the pass/fail test goes blind (this is the "masking" half).
  - *On the continuous lens:* agreement is near-uniform but leaves a **tiny age-correlated
    residual** (60+ ≈0.971 vs <40 ≈0.978), and the collapsed variance turns that ≈0.6-pt
    residual into the run's **strongest** FDR-significant signal. Silver inflates the gap's
    **certainty**, not its size.

So our setup demonstrates a **distinct, mirror-image** biased-ruler mode (false confidence
via variance collapse). What it *cannot* easily show is the binarized-DIR **magnitude**
inflation — for **two** reasons, both worth stating before spending compute:
1. the twin saturation above, and
2. the whole task sits far above the 0.8 bar — the **gold** ruler also nearly saturates
   (DIRs 0.94–1.0, 0/63), so there is little headroom for *any* ruler to push a subgroup
   below 0.8.

This is partly a genuine limitation of our setup (CSpineSeg images carry *either* gold *or*
silver, never both, forcing us to *generate* the second ruler with a too-similar model) and
partly a property of the high-Dice regime.

## A strength, not just a caveat: the age effect is intrinsic

The age gradient (**60+ worst**) shows up against **both** rulers *and* in the encoder
probe (age decodable above the random-init null, race at the null;
[`../demographic-probing-of-medical-image-encoders/findings.md`](../demographic-probing-of-medical-image-encoders/findings.md)).
An effect visible through the true ruler, the correlated ruler, *and* the representation is
unlikely to be a label artifact — it is **intrinsic/representational difficulty**, directly
corroborating Parikh et al.'s Exp-2/3 conclusion (the bias survives label swaps and
difficulty balancing). This "survives every lens" point is evidence the effect is *real*,
not part of the limitation.

> **Data caveat (hedge before a sharp reader catches it).** "60+ worst" is robust — lowest
> on both mean and median, on both rulers. But the gold gradient is **not cleanly monotone
> by mean**: 40–60 (0.900) > <40 (0.896) > 60+ (0.894), because the **<40 group is
> high-variance** (std 0.098 vs ~0.03) with a fat lower tail. The clean "young best → old
> worst" ordering holds on **medians** (which the rank-based Kruskal test reads); the silver
> ruler *is* cleanly monotone (means 0.978 → 0.972 → 0.971). State "60+ worst," not a
> strict monotone decline.

## Is the model actually *unfair*? No — significance ≠ disparity

A statistically significant gap is **not** the same as an unfair model, and we must not
pitch it as one. Significance answers *"is the difference reliably non-zero?"*; unfairness
answers *"is it large enough, and harmful enough, to matter?"* Here the first is yes
(against silver) and the second is **no**:

- **The magnitudes are clinically negligible.** Worst-vs-best age group is **0.900 → 0.894**
  on gold (a **0.6%** macro-Dice gap) and **0.978 → 0.971** on silver (**0.7%**). Every
  patient gets a ~0.9+ Dice segmentation; no subgroup is failed.
- **It clears the field's own adverse-impact line by a wide margin.** Gold age DIR = **0.95**,
  global-audit DIRs **≥ 0.97** — all far above the four-fifths **0.80** threshold used to
  *declare* adverse impact. By the operational definition fairness research uses, this is fair.
- **There isn't even a clean victim.** The group with the fattest lower tail — the actual
  catastrophic cases — is **`<40`** (gold std 0.098, worst cases ≈0.80), *not* 60+. The
  "old worst" trend holds only on the mean/median and barely; ask Parikh et al.'s literal
  question — *"who does your algorithm fail?"* — and the worst individual cases are a few
  **young** outliers. "Biased against older patients" is therefore **not** a defensible
  headline.

**The significance is an artifact of the ruler, not evidence of harm.** The silver ruler's
variance collapse turns a clinically irrelevant 0.6–0.7% gap into a "significant" one — see
`age_trend_gold_vs_silver.png` (this run dir): identical downward staircase on both rulers,
but gold's boxes are tall/noisy → n.s., silver's are razor-thin → significant. That is the
whole point and the whole caution: **a low-noise / auto-generated reference label can
manufacture a finding out of a non-issue** — the mirror image of Parikh's biased ruler
inflating a *real* gap.

**So the verdict is "fair," and the contribution is methodological, not a disparity.** Pitch
the audit as *fair across sex, race, and age (sub-percent gaps, DIR ≫ 0.8)*; pitch the
biased-ruler experiment as *be careful what you call unfair* (ruler choice flips the
significance verdict on a negligible effect). The only place a *real* fairness story could
still emerge is the **amplification** experiment (does *training* on silver labels widen the
gap, cf. Parikh Exp 4) — a different question from "is the gap significant," and the one
worth the compute.

## Options

1. **Keep the current analysis, reframed as the verdict-flip / false-confidence mode.**
   Report that the same model is age-fair against gold (0/63) but age-unfair against silver
   (11/63) — a biased-ruler effect via variance collapse, the mirror of Parikh's magnitude
   inflation — plus the intrinsic-age evidence above. State the twin limitation plainly.
   Valid and complementary to Parikh et al. as-is.
2. **Finish the bias-amplification experiment (Dataset002 gold vs Dataset003 silver).**
   Already in motion (DS003 training), no external model, no label-mismatch risk, and it
   maps directly onto Parikh et al.'s strongest result (Exp 4). Answers the deeper
   question — does silver *training* hurt fairness — not just the ruler question.
3. **Reproduce the magnitude-inflation mode with a deliberately independent/biased in-house
   silver generator.** Retrain a silver generator on a split **disjoint** from DS001's
   training data (and/or weaker config, fewer cases, age-skewed subset). Lower-risk than
   importing an external model (same label space, full control) and scientifically cleaner
   — you *manipulate* the silver bias and can show the ruler effect scale with it. **Caveat:**
   in a ~0.97-Dice regime the binarized DIR only moves if the injected bias is strong enough
   to push a subgroup **below the 0.8 bar** — so this only rescues *DIR* inflation if the
   silver is made substantially, not subtly, biased.
4. **Import an external spine segmentation model** (e.g. TotalSpineSeg, already in the
   probing lineup) as the silver generator. Most faithful to "real-world independent
   annotator," but highest cost/risk: label-taxonomy remapping to our VB/disc cervical
   classes, and provenance/leakage checks (the same reason we rejected reusing Zhou et
   al.'s original silver model). Same below-0.8 caveat as Option 3.

## Recommendation

- **Backbone:** keep the current biased-ruler result, reframed as the **verdict-flip /
  false-confidence** mode (Option 1) — we reproduce Parikh's core thesis via a complementary
  mechanism; foreground that plus the intrinsic-age evidence.
- **Priority next step:** finish **bias amplification** (Option 2) — it is the real,
  low-cost, low-risk comparison to Parikh et al. and needs no new model.
- **Only if the ruler experiment must also show magnitude inflation:** prefer a **controlled
  biased in-house generator** (Option 3) over an external model (Option 4) — but note the
  below-0.8 caveat: in this high-Dice regime even an independent biased silver may not move
  the binarized DIR unless the injected bias is strong.
- **Raise with the supervisor first** — he authored the biased-ruler papers; framing the
  choice as "we reproduce the verdict flip via variance collapse (false confidence), not
  magnitude inflation, because our silver is a correlated twin and the task sits above the
  0.8 bar; lean on amplification, or build a strongly-biased independent silver to also show
  magnitude inflation?" is cheap to ask before spending compute.

## Related

- [`fairness-runs.md`](fairness-runs.md) — Run 8 (biased ruler), Run 9 (amplification — done, no amplification)
- [`dpd-dir-redefinition.md`](dpd-dir-redefinition.md) — why the silver ruler saturates under the binarized DIR
- [`related-work.md`](related-work.md) — novelty positioning
- [`../nnunet/06_gold_silver_training.md`](../nnunet/06_gold_silver_training.md) — dataset/label flow, amplification design
- [`../demographic-probing-of-medical-image-encoders/findings.md`](../demographic-probing-of-medical-image-encoders/findings.md) — encoder probe corroborating age as the salient axis
