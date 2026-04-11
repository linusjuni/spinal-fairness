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
| Test | 113 | 113 | 50.0% |





# Notes from meeting

- ~~Actually we should balance EXACTLY. So we might have to drop some exams for the multiple patients.~~ ✓ Done — v3 now balances at exam level.

- we should do statistical tests to compare means

- make a really good table where we compare different things from the eda in a well structured manner

- aditya will find some libraries that can be used to check segmentations (rounded edges, square, etc)

- do the whole eda on the training set only too (potentially also on val and test just to see what it looks like)

- do a much bigger deep dive into the volumentric analysis
