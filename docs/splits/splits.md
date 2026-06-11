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
| Train | 399 | 399 | 50.0% |
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
gold test set (split_v3_gold) isolates the effect of silver training labels on
fairness gaps (bias amplification experiment).

---

## Dataset composition summary (verified 2026-06-05)

All counts verified from the authoritative TSV files; `annotation_quality` column
in `split_v3.tsv` is the ground truth for gold/silver membership.

### Per-split case counts

| Dataset | Split file | Train | Val | Test | Total |
|---------|-----------|-------|-----|------|-------|
| Dataset001 (Mixed) | split_v3 | 798 (318 gold + 480 silver) | 116 | 228 | 1,142 |
| Dataset002 (Gold) | split_v3_gold | 288 | 44 | 76 | 408 |
| Dataset003 (Silver) | split_v3_silver | 450 | 66 | 138 | 654 |

All splits are exactly 50/50 M/F (verified per-split, per-file).

### Why Dataset001 gold-train (318) ≠ Dataset002 train (288)

The two split files were created with independent sex-stratified downsampling runs.
`split_v3_gold` is a stricter sex-balanced resample of the full gold pool: 30 gold-train
cases that appear in `split_v3` are entirely absent from `split_v3_gold` (they were dropped
when v3_gold was resampled). The 288 cases in `split_v3_gold` train are a strict subset of
the 318 gold-train cases in `split_v3`. This is intentional, not an inconsistency.

---

## Test set composition

The full test set (split_v3, 228 cases) breaks down as follows:

```
228 total test cases
├── 76  in labelsTs_gold  (expert labels → Dataset002 evaluation reference)
├── 138 in labelsTs_silver (auto-generated labels)
└── 14  in neither labelsTs_gold nor labelsTs_silver
    (silver test cases whose labelsTs symlinks were not created)
```

76 + 138 + 14 = 228 ✓. The `labelsTs/` directory in Dataset001 contains all 228.

The 14 cases present in `split_v3` test but absent from both `labelsTs_gold` and
`labelsTs_silver`:

```
cspine_000057   cspine_000134   cspine_000314   cspine_000364
cspine_000384   cspine_000435   cspine_000459   cspine_000495
cspine_000524   cspine_000659   cspine_000886   cspine_000980
cspine_001033   cspine_001088
```

These are silver-annotated test cases that were not symlinked into the
`labelsTs_gold` or `labelsTs_silver` subdirectories.

### Biased ruler: predictions_test_pp vs labelsTs_gold

`Dataset002_CSpineSeg_Gold/predictions_test_pp/` contains 76 `.nii.gz` prediction
files (plus `eval_gold.csv`). The 76 case_ids are identical to `labelsTs_gold/`.
This confirms the same 76 images are scored under both the gold (expert) and
generated-silver (Dataset002 predictions) rulers. Verified 2026-06-05.
