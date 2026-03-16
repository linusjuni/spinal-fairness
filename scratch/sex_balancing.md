# Sex Balancing for Fairness Audit

The dataset has a mild sex imbalance (~55% female, 45% male), reflecting the
underlying clinical population. For the sex-based fairness audit, equal
representation across groups is desirable. Three split versions were explored.

---

## split_v1 — Baseline (race × age stratification only)

No sex control. Splits are stratified by race_bin × age_bin.

| Split | Female | Male | F% |
|-------|--------|------|----|
| Train | 476 | 400 | 54.3% |
| Val | 68 | 58 | 54.0% |
| Test | 139 | 113 | 55.2% |

The ~55/45 imbalance is consistent across all splits, reflecting the dataset.

---

## split_v2 — Sex added to stratification key (race × age × sex)

Sex is added as a stratification axis alongside race_bin and age_bin.

| Split | Female | Male | F% |
|-------|--------|------|----|
| Train | 478 | 398 | 54.6% |
| Val | 67 | 57 | 54.0% |
| Test | 138 | 116 | 54.3% |

Stratification ensures proportional consistency *across* splits but cannot
change the underlying population ratio. The result is virtually identical to
v1. Restratification is not the right tool for sex balancing.

---

## split_v3 — Full downsample to 50/50 across all splits

Following Aditya et al. (MAMA-MIA), female patients are downsampled in every
split to match the male count in that split. Dropped patients are excluded
entirely. This produces a fully balanced cohort for both training and
evaluation, enabling controlled experiments to test whether representational
imbalance causes observed fairness gaps.

Dropped: 89 train / 11 val / 17 test female patients (1254 → 1136 exams total).

| Split | Female | Male | F% |
|-------|--------|------|----|
| Train | 393 | 395 | 49.9% |
| Val | 56 | 58 | 49.1% |
| Test | 116 | 118 | 49.6% |

### Caveat: patient-level balancing vs exam-level counts

Balancing is performed at the **patient level** to prevent data leakage from
the 23 multi-exam patients. Because these patients are not evenly distributed
by sex across splits, equal patient counts do not guarantee perfectly equal
exam counts. This produces a residual imbalance of 1–2 exams per split (worst
case: val 56F vs 58M). This is considered acceptable — the discrepancy is too
small to meaningfully affect DPD or DIR metrics.
