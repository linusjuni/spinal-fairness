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

Following Aditya et al. (MAMA-MIA), female exams are downsampled in every
split until the female exam count exactly matches the male exam count. This
produces a fully balanced cohort for both training and evaluation, enabling
controlled experiments to test whether representational imbalance causes
observed fairness gaps.

Balancing operates at the **exam level**: female exams to drop are selected
uniformly at random (seeded for reproducibility). For the 23 multi-exam
patients, some exams may be retained while others are dropped — this is safe
because v2 already confines all exams from a patient to the same split, so
no leakage is introduced.

Dropped: 80 train / 7 val / 25 female exams (1254 → 1142 exams total).

| Split | Female | Male | F% |
|-------|--------|------|----|
| Train | 400 | 400 | 50.0% |
| Val | 58 | 58 | 50.0% |
| Test | 114 | 114 | 50.0% |

---

## split_v3_gold — Sex-balanced gold-only subset

Filters split_v3 to `annotation_quality == "gold"` (expert-annotated cases), then
re-applies sex-balancing within each split. Because gold cases skew slightly female
in train but male in val/test, the generalised balancer drops the majority sex in
each split independently.

Dropped: 30 Female from train, 3 Male from val, 7 Male from test.

| Split | Female | Male | F% |
|-------|--------|------|----|
| Train | 144 | 144 | 50.0% |
| Val | 22 | 22 | 50.0% |
| Test | 38 | 38 | 50.0% |

Used for Dataset002_CSpineSeg_Gold training. The test set (76 cases) is the
ground-truth reference for the gold vs silver label comparison — both models
should be evaluated against it.

---

## split_v3_silver — Sex-balanced silver-only subset

Filters split_v3 to `annotation_quality == "silver"` (auto-generated labels), then
re-applies sex-balancing within each split.

Dropped: 30 Male from train, 3 Female from val, 7 Female from test.

| Split | Female | Male | F% |
|-------|--------|------|----|
| Train | 225 | 225 | 50.0% |
| Val | 33 | 33 | 50.0% |
| Test | 69 | 69 | 50.0% |

Used for Dataset003_CSpineSeg_Silver training. Evaluating this model against the
gold test set (split_v3_gold) gives the true fairness gap. Evaluating against the
silver test set reveals the Biased Ruler effect.
