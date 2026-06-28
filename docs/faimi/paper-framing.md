# FAIMI Paper Framing & Positioning

How to frame the `submission/` paper for FAIMI, distilled from supervisor
guidance (Aasa Feragen, Aditya Parikh) following Linus's group presentation and
the Slack thread around Aasa's letter to the CSpineSeg authors (Evan et al.).

**Decision:** take **Aasa's advice** on what the story is (lead with the
validation / biased-ruler problem), **keep Aditya's "first on spinal" claim**
prominently, and **frame the whole thing as an extension of Parikh et al.'s
biased-ruler work** — we did not invent that effect, Aditya did. These are
compatible — see [Reconciled framing](#reconciled-framing).

---

## Intellectual lineage: the biased ruler is Parikh et al.'s, we extend it

The biased-ruler effect is **not our contribution to coin** — it originates in our
own supervisors' prior breast-MRI work, and our framing must credit it as the
lineage we build on, not as something we discovered.

### `aditya1` — *Investigating Label Bias and Representational Sources of Age-Related Disparities in Medical Segmentation* (Parikh, Das, Feragen)

This is the paper that **introduces the "Biased Ruler" effect for segmentation.**
In its own words: *"It is known from classification tasks that this gives a biased
ruler effect, where we are unable to effectively measure and mitigate bias. In
this paper, we therefore let machine-generated labels from a pre-existing system
serve as a methodological probe... to quantify the impact of using a flawed,
real-world label."* On MAMA-MIA (breast DCE-MRI) it establishes:

- **The biased-ruler (evaluation) effect = "false magnitude."** Scoring against
  biased silver labels *inflated observed bias by ~40%* over the true gap
  (fairness gap 0.0399 true → 0.0559 observed; DIR **0.871 → 0.815**). The flawed
  ruler makes the gap look *bigger* than it is.
- **Bias amplification (training) effect.** Training on biased silver labels
  *widened* the fairness gap by **66%** (0.0399 → 0.0661), pushing DIR below the
  four-fifths line to **0.79**. "Systemic bias is learned and propagated."
- **Intrinsic difficulty.** Younger patients are intrinsically harder (tumours
  66% larger, 70% more variable); the bias survives training-set balancing and
  difficulty-balancing — it is not merely label noise or case imbalance.

### `aditya2` — *Who Does Your Algorithm Fail? Investigating Age and Ethnic Bias in the MAMA-MIA Dataset* (Parikh, Das, Feragen)

This originates the **auditing framework we adopt wholesale**: the *fairness under
unawareness* paradigm, the **DPD / DIR** metrics, the **four-fifths (0.8) rule**,
the **top-25% beneficial-outcome** binarization, balance-by-downsampling, and
OLS/Kruskal–Wallis confounder control. Findings relevant to us: age bias is
**intrinsic** (persists after controlling for data source), and ethnic bias can
be **masked by data aggregation** (Asian HD95 DIR 0.52 hidden in the pooled
analysis) — a site-confounding problem that does *not* arise for us (CSpineSeg is
single-institution).

### What is genuinely ours, relative to Parikh et al.

| | Parikh et al. (breast MRI) | This work (cervical spine) |
|---|---|---|
| Ruler type | **independent**, itself-biased model | **correlated** near-twin of the model under test |
| Ruler failure mode | **false magnitude** — inflates the gap (DIR 0.871→0.815) | **false confidence** — collapses within-group variance ~4×, flips a significance verdict with no real gap |
| Training on silver | **amplifies** bias (+66%, DIR→0.79) | **no amplification** (silver matches/beats mixed) |
| Anatomy | breast (first label-bias audit there) | cervical spine (**first fairness audit of this anatomy**) |

So our three real contributions all sit **inside Parikh et al.'s framework**:
(1) extend it to a new anatomy; (2) discover a *complementary mode* of the
biased-ruler effect (false confidence) that arises when the ruler is correlated
rather than independently biased; (3) find the boundary condition where
amplification *does not* occur. The novelty is the **mode and the setting**, not
the effect.

---

## What the supervisors said

### Aasa — the story must be the validation problem, not the new anatomy

Aasa's strategic steer (her FAIMI question to Linus):

> "I am not sure that assessing bias on a new type of dataset will necessarily be
> enough, but the validation issues and the way they affect fairness auditing are
> interesting."

Read as: **a fairness audit of a new anatomy, on its own, is not a strong enough
FAIMI contribution. The methodological finding — that reference-label provenance
distorts fairness auditing — is.** That should be the headline.

She has independently validated and escalated this finding. In her letter to the
CSpineSeg authors she reframes our biased-ruler result as a **community-level
data-leakage / performance-overestimation problem**:

- The dataset's silver labels were produced by a model trained on its gold
  labels, so silver labels **carry information about the gold-labelled images**.
- Any new model trained (partly) on the same gold labels will agree more with
  the silver labels than with ground truth — **two models trained on overlapping
  data resemble each other more than they resemble the truth.**
- Consequence: **users who score against silver labels overestimate
  performance** (we measured ~8 Dice points: 0.897 gold vs 0.977 silver on the
  same model). The only safe fix is to report on **gold** test labels — which
  requires users to know *which labels are which* and to be aware of the issue.

She considers this "sufficient... that the community should be made aware," and
wants to publish the warning *with* the dataset authors. That makes the
validation angle the part of our work with the clearest external traction.

### Aditya — claim the "first spinal fairness audit," and ship now

Aditya's guidance, two parts:

1. **Strongly encourages the "first work on cervical-spine fairness" claim.** It
   is a legitimate novelty hook and should stay prominent — not buried.
2. **Ship it.** Don't delay for a better story or better results:

   > "Delaying such projects in hope for better story or results, where you have
   > constraints due to data, and results which we didn't expect — often get lost
   > and nobody later has will to finish it."

   Workshop best case, arXiv worst case; both are good. He has offered to help
   finish/polish if Linus runs out of time. The two `null` results (fair model,
   no amplification) are fine — they are *constraints of the data*, not a failure.

---

## Reconciled framing

Three steers, not two, and they nest cleanly once we separate **lineage**,
**headline contribution**, and **novelty hook**:

- **Lineage (Parikh et al. / Aditya's prior work):** the biased-ruler effect and
  the DPD/DIR auditing framework are theirs. We *extend* them — explicitly,
  prominently, in the intro and related work. This is not just citation hygiene:
  positioning our finding as a *new mode of a known effect* is stronger and more
  honest than implying we found the effect. (Bonus: two of the three FAIMI
  reviewers' natural reference points — and our own co-authors — are these
  papers.)
- **Headline contribution (Aasa):** *Reference-label provenance is a first-order
  confounder in segmentation fairness auditing.* It both (a) inflates raw
  performance (~8 Dice points, leakage-driven) and (b) flips fairness verdicts —
  the **"false confidence"** / variance-collapse mode, **complementary to Parikh
  et al.'s "false magnitude."** Plus a practitioner diagnostic for detecting an
  over-correlated ruler.
- **Novelty hook / vehicle (Aditya):** *The first demographic fairness audit of
  cervical-spine MRI segmentation* (sex, race, age) is the controlled setting
  that lets us isolate the ruler effect. We keep the "first" claim, but it frames
  the audit as the **vehicle** that produces the methodological finding rather
  than the headline result on its own.

The narrative in one breath: **Parikh et al. showed a biased silver *ruler* can
inflate apparent bias (false magnitude) and that training on silver amplifies it.
We carry their framework to cervical spine — the first fairness audit of this
anatomy — and find a complementary failure mode: a *correlated* silver ruler
manufactures *false confidence* (it collapses within-group variance and flips a
significance verdict with no real disparity), while training on silver does *not*
amplify here. Same effect family, opposite mechanism, new anatomy — and a
concrete warning for anyone validating on a mixed-label dataset.**

One-line version: **"The first fairness audit of cervical-spine MRI segmentation
— and a complementary mode of Parikh et al.'s biased-ruler effect: when the ruler
is a twin of the model, it doesn't exaggerate bias, it manufactures false
confidence."**

This states "first" up front (Aditya), centers the validation/ruler story (Aasa),
and credits the biased-ruler lineage (Parikh et al.) as the thing we extend.

---

## Concrete implications for the paper

### Contributions list (rewrite emphasis, `introduction.tex`)

Current order presents four co-equal contributions with the audit (#1) reading as
the headline. Re-rank so the methodological finding leads, keep "first" as the
opening novelty hook, and **make the Parikh-et-al. lineage explicit in the framing
sentence that precedes the list** (e.g. "Building on the biased-ruler framework of
Parikh et al. [cite aditya1, aditya2], we ..."):

1. **(Hook)** First demographic fairness audit of cervical-spine MRI segmentation
   across sex, race, and age — fair by every standard metric.
2. **(Headline)** A **complementary mode of the biased-ruler effect of Parikh et
   al.**: swapping expert for machine-generated reference labels flips age from
   non-significant to significant — not by inflating the gap (their "false
   magnitude") but by collapsing within-group variance (**"false confidence"**).
   The two modes correspond to whether the silver ruler is *independently biased*
   (theirs) or *correlated with the model under test* (ours).
3. **(Headline support)** The same mechanism inflates **raw** performance by
   ~8 Dice points (gold vs silver ruler on identical predictions) — a
   leakage-driven overestimation that affects *any* user of a mixed-label
   dataset, not just fairness auditors.
4. **Boundary condition on amplification:** training on silver labels does **not**
   amplify bias here, in contrast to the +66% amplification Parikh et al. report
   in breast MRI — and encoding ≠ disparity (sex strongly encoded, zero
   performance gap).
5. **(Actionable)** A diagnostic for an over-correlated ruler: scores much higher
   *and* much tighter against the ruler than against expert labels (~4× variance
   collapse) signal the ruler is too close to the model to be an independent
   benchmark.

### Abstract / intro

- Keep the "first cervical-spine fairness audit" sentence (Aditya).
- Shift the abstract's weight toward the ruler/validation finding and the
  ~8-point inflation, so a reader sees the methodological contribution as the
  point — not the (null) fair verdict.
- Promote the **practitioner diagnostic** out of future-work into a named
  contribution (it's the transferable, generalizable artifact FAIMI rewards).

### The leakage angle is now first-class

The ~8-point inflation (raw performance, not just fairness) is the version of our
result Aasa is escalating to the dataset authors. The paper already reports it
(`experiments.tex`, §Biased Ruler: 0.897 → 0.973). **Surface it earlier and more
prominently** — it generalizes beyond fairness to anyone validating on this
dataset, which widens the audience.

#### Aasa's letter to the CSpineSeg authors (2026-06-28) — mechanism is *leakage*

After Linus's group presentation, Aasa wrote to Evan et al. stating the mechanism
precisely, and we now **follow this story entirely**:

- Cause of the offset is **label leakage**: the dataset's silver labels were generated
  by a model trained on its gold labels, so silver "already carries information about
  the human-labelled images." Any new model trained (partly) on the same gold labels
  then agrees more with the silver labels than with truth — *"the outputs of two models
  trained on (in part) the same data are likely to be more similar to each other than to
  the underlying true or human-labelled segmentations."*
- Her headline to the authors is the **offset, not the age dependency** (she explicitly
  says the age trend "looks similar across label types" — matching our both-rulers
  finding — and that the *interesting* thing is the performance offset). So in the
  paper the **offset/overestimation gets co-equal-or-leading billing** with the
  false-confidence verdict-flip.
- Her fix: **report on gold (expert) test labels**, which presupposes users know which
  labels are which. Now recommendation **(1)** in `discussion.tex`.
- She wants to publish the warning **with** the dataset authors — external traction, and
  confirmation that the leakage angle is the load-bearing contribution.

**Mechanism resolution (was ambiguous, now settled).** The decision note
[`../fairness/biased-ruler-silver-source.md`](../fairness/biased-ruler-silver-source.md)
had two co-equal hypotheses: H1 (leakage / correlated errors) and H2 (task simplicity).
Aasa's letter resolves this: **H1/leakage is primary; H2/task simplicity is a compounding
amplifier.** This corrected a real internal inconsistency — `experiments.tex` already said
"leakage," but `discussion.tex` had attributed the offset to "the inherent simplicity of
the task" with a footnote claiming the $M_{\text{gold}}$/$M_{\text{silver}}$ disjoint-data
agreement proved it was "a property of the task, not the training overlap." That inference
was wrong: the images are disjoint but $M_{\text{silver}}$'s *labels* are downstream of
gold, so that 0.97 agreement is **evidence for** pervasive label leakage, not against it.
The footnote is removed and folded into the main text as exactly that evidence.

#### Status of the reframe (2026-06-28)

Done in `submission/`: `experiments.tex` (E2 retitled "Performance Inflation and False
Confidence"; leakage mechanism named), `discussion.tex` (leakage-primary "Two modes" para;
report-against-gold added as recommendation 1), and the three previously-skeleton sections
now **drafted leakage-forward** — `abstract.tex`, `introduction.tex` (contributions list
filled: #2 = leakage/overestimation, #3 = false-confidence mode), `conclusion.tex`.

**8pp scope cut (round 3):** the **E3/E4 amplification thread was removed** as the
least-Aasa-supporting content (see `content-cut-from-paper.md`). The submission is now
audit (E1) + biased-ruler/leakage (E2); contribution #4 (boundary condition on
amplification) is gone, leaving 4 contributions. `M_silver` is retained only as leakage
evidence. Amplification is reinstatable from `paper/` for a longer venue — **flag to
Aditya**, whose boundary-condition result this was.
Still pending: anonymize `main.tex`; rebuild + check 8pp budget (no local LaTeX toolchain).

---

## Risk to shore up before writing the story

The causal claim "the ~8-point gap is a *label-type* effect, not an
*image-difficulty* effect" rests entirely on **Zhou et al. assigning gold/silver
pseudo-randomly (by medical-record number, no demographic/difficulty
stratification)**. This single assumption carries both Aasa's letter and our
headline number.

- It is already cited (`methodology.tex:218`, `zhou2025cspineseg`).
- A reviewer's first objection will be *"maybe the gold cases were just harder."*
- **Back it empirically:** one line showing the gold and silver pools match on
  age/sex/race marginals (and ideally a difficulty proxy), so comparability is
  shown, not just asserted.

---

## Logistics reminder

- Hard limits live in [`submission-requirements.md`](submission-requirements.md):
  **8 pages + 2 for references, LNCS, double-blind.** The reframing must happen
  *within* 8 pages — promoting the ruler story is also a chance to **cut** the
  methodology/discussion bulk that currently overflows.
- Ship target per Aditya: **FAIMI workshop best case, arXiv worst case.** Don't
  hold for a better result.
