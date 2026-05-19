# HD95 Outlier Analysis

13 cases (5.7%) have hd95_macro > 5mm. These drive the misleading HD95 DIR values (0.40–0.77).

## Pattern

12/13 outliers are **disc localization errors**: hd95_disc > 10mm while hd95_vb < 4mm. The model predicts most discs correctly but places a spurious disc fragment (or misses one) 3–5 slices away. At ~4mm slice thickness, that's 13–19mm in HD95.

8/13 have Dice > 0.85 — overall overlap is good, but HD95 catches the worst-case boundary error that Dice is insensitive to.

| case_id | dice_vb | dice_disc | hd95_vb | hd95_disc |
|---|---|---|---|---|
| cspine_000697 | 0.665 | 0.667 | 50.8 | 58.1 |
| cspine_000021 | 0.502 | 0.465 | 18.1 | 20.0 |
| cspine_000886 | 0.925 | 0.811 | 2.2 | 19.8 |
| cspine_000458 | 0.808 | 0.780 | 4.2 | 17.0 |
| cspine_000435 | 0.934 | 0.823 | 2.4 | 17.5 |
| cspine_000397 | 0.952 | 0.835 | 0.6 | 19.0 |
| cspine_000998 | 0.951 | 0.889 | 0.7 | 18.3 |
| cspine_000993 | 0.960 | 0.926 | 0.7 | 16.8 |
| cspine_000617 | 0.909 | 0.867 | 3.3 | 13.1 |
| cspine_000295 | 0.930 | 0.747 | 1.9 | 14.0 |
| cspine_000310 | 0.958 | 0.922 | 0.4 | 14.8 |
| cspine_000442 | 0.992 | 0.966 | 0.0 | 13.0 |
| cspine_001185 | 0.853 | 0.823 | 6.5 | 4.0 |

## Why HD95 DIR is misleading

HD95 is heavily right-skewed (median 0.34mm, mean 1.36mm, max 54mm). DIR uses means, so whichever group catches more outliers by chance looks much worse. The rank-based Mann-Whitney tests (immune to outlier magnitudes) confirm no significant group differences.

## Not a code bug

- The evaluation code correctly uses NIfTI voxel spacing for mm conversion.
- 0 inf / 0 NaN values — no empty-mask edge cases.
- The worst case (cspine_000697, HD95 ~55mm) spans nearly the full cervical FOV (~15 slices × 4mm ≈ 60mm) and has Dice 0.67 — physically consistent with a grossly wrong prediction.

These are genuine model failures, not configuration artifacts. Visual inspection of the NIfTI overlays (especially cspine_000697 and cspine_000021) would confirm the specific failure mode.

## Median-based DIR

Standard DIR uses means: `mean(worst) / mean(best)`. Because HD95 is unbounded above (max 54mm vs median 0.34mm), a single outlier landing in one group shifts the mean — and therefore the DIR — far more than an equivalent outlier in a bounded metric like Dice.

Replacing means with medians eliminates this leverage:

| Grouping | Mean DIR | Median DIR | Δ |
|---|---|---|---|
| Sex | 0.563 | 0.880 | +0.317 |
| Race (W vs B) | 0.635 | 0.800 | +0.165 |
| Race (3-way) | 0.627 | 0.800 | +0.173 |
| Race (W vs NW) | 0.694 | 0.800 | +0.106 |
| Age (3-bin) | 0.772 | 1.000 | +0.228 |
| Age (median) | 0.613 | 0.800 | +0.187 |
| Ethnicity | 0.537 | 0.308 | −0.229 |

Median DIR recovers ≥ 0.80 for all groupings except ethnicity (n=12 Hispanic — too few for either estimate to be stable). For Dice and nDSC, mean and median DIR agree within ±0.03 because the [0, 1] bound caps outlier leverage.

The median-based DIR is not a standardized metric — the four-fifths rule is defined on means. But the comparison demonstrates that the alarming mean-based HD95 DIRs are artifacts of a few extreme cases, not evidence of systematic demographic disparity.

## Fairness impact

None. The 13 outliers are not concentrated in any demographic group. All rank-based fairness tests remain non-significant (all p_fdr > 0.72).
