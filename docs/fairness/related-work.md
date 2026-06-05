# Related Work & Novelty Positioning

Where this study sits relative to existing fairness literature. Based on a
deep literature sweep (2026-06-05). Bottom line: **we are not the only fairness
study touching spine segmentation, but our scope is novel.** The "first work on
fairness in spine segmentation" claim must be scoped — see below.

## The one prior work on spine-segmentation fairness

**FairMedFM** — Jin et al., *NeurIPS 2024 Datasets & Benchmarks*,
[arXiv:2407.00983](https://arxiv.org/abs/2407.00983). A broad fairness benchmark
for medical-imaging foundation models (20 models × 17 datasets). One of those
datasets is **SPIDER** (218 3D *lumbar* spine MRI, vertebra + disc masks), which
FairMedFM evaluates for **segmentation** fairness across **sex** (ΔDSC, DSC
skewness). This is the same task + modality + anatomy family as ours, so an
unqualified "first fairness work on spine segmentation" is **not defensible**.

But the precedent is shallow, and we differ on four axes — each genuinely first:

| Axis | FairMedFM (SPIDER) | This work |
|---|---|---|
| Attributes | sex only | **sex + race + age** |
| Region | lumbar | **cervical** |
| Depth | one dataset in a 17-dataset sweep | **dedicated** spine study |
| Label bias | none | **gold-vs-silver "biased ruler"** |

**Defensible claim:** *the first dedicated demographic fairness audit of
cervical-spine MRI segmentation across sex, race, and age, and the first to study
label bias in spine segmentation via a gold-vs-silver "biased ruler" analysis* —
citing FairMedFM explicitly as the prior (sex-only, lumbar, benchmark-level) work.

## Distinctions to keep sharp in related work

- **Gichoya et al. 2022** (*Lancet Digital Health*) used cervical-spine
  *radiographs* only to **predict** race (AUC 0.913) — not to segment.
  Race-prediction-from-spine-images ≠ fairness-of-spine-segmentation. Cite as
  motivation, not as a competing claim.
- **Fairness-in-segmentation** precedents are **cardiac** (Puyol-Antón et al.
  2022 — "first racial bias in cine CMR segmentation") and **brain** (Danaee et
  al. 2025, [arXiv:2510.17999](https://arxiv.org/abs/2510.17999), nucleus
  accumbens, race & sex). Spine is otherwise the gap. The "first in [organ]
  segmentation" framing is accepted practice in this literature.
- Other **spine-AI fairness** work is tabular/EHR (surgical-outcome, scoliosis,
  spinal-fusion disparities) or detection/classification — none is spine
  *segmentation* fairness.

## Pre-submission checks

- Confirm SPIDER lacks race/age labels (supports our race/age novelty on spine).
- Final 2026-preprint sweep for any cervical-spine segmentation fairness work.
