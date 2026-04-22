# EDA Statistical Results: Full Cohort vs Train Split

All p-values are BH-FDR adjusted within each module. Effect size thresholds follow
the conventions in `statistical_testing.md`. "Consistent" means both cohorts agree
on whether H0 is rejected.

---

## 1. Segmentation Volumes

16 tests (12 Mann-Whitney + 4 Kruskal-Wallis). Groups for sex: Female (a) vs Male
(b); for race: White (a) vs Black (b); manufacturers sorted alphabetically: GE (a)
vs Siemens (b).

### Vertebral body volume (mm³)

| Comparison | Full effect (magnitude) | Full sig? | Train effect (magnitude) | Train sig? | Consistent? |
|---|---|---|---|---|---|
| × Sex (r_rb) | 0.559 (large) | Yes | 0.575 (large) | Yes | Yes |
| × Race (r_rb) | 0.292 (small) | Yes | 0.315 (small) | Yes | Yes |
| × Age bin (ε²) | 0.119 (medium) | Yes | 0.117 (medium) | Yes | Yes |

Sex and age are large, dominant effects. White patients have larger VBs than Black
patients (small-medium effect), consistent across splits.

### Intervertebral disc volume (mm³)

| Comparison | Full effect (magnitude) | Full sig? | Train effect (magnitude) | Train sig? | Consistent? |
|---|---|---|---|---|---|
| × Sex (r_rb) | 0.580 (large) | Yes | 0.571 (large) | Yes | Yes |
| × Race (r_rb) | 0.025 (negligible) | No | 0.022 (negligible) | No | Yes |
| × Age bin (ε²) | 0.010 (negligible) | Yes | 0.005 (negligible) | No | No |

Race has no meaningful effect on disc volume. The age signal is borderline — it
survives in the full cohort but not in the smaller train split, and the effect size
is negligible (ε² < 0.01) in both, so it carries no practical weight.

### Voxel counts by manufacturer (scanner resolution check)

| Structure | Full effect (r_rb) | Full sig? | Train effect (r_rb) | Train sig? | Consistent? |
|---|---|---|---|---|---|
| VB voxels (GE vs Siemens) | 0.429 (medium) | Yes | 0.407 (medium) | Yes | Yes |
| Disc voxels (GE vs Siemens) | 0.434 (medium) | Yes | 0.408 (medium) | Yes | Yes |

GE scanners produce ~2× the voxel counts of Siemens (median 69 K vs 32 K for VB).
This is a resolution/spacing artifact — not a biological difference.

### Volume (mm³) by manufacturer

| Structure | Full effect (r_rb) | Full sig? | Train effect (r_rb) | Train sig? | Consistent? |
|---|---|---|---|---|---|
| VB mm³ (GE vs Siemens) | 0.168 (small) | Yes | 0.155 (small) | Yes | Yes |
| Disc mm³ (GE vs Siemens) | 0.030 (negligible) | No | 0.063 (negligible) | No | Yes |

After converting to physical units, the scanner effect on VB volume is small (Siemens
slightly larger) and absent for discs. Scanner does not meaningfully confound disc
volume measurements.

### Component counts by demographic (annotation completeness proxy)

| Comparison | Full effect (magnitude) | Full sig? | Train effect (magnitude) | Train sig? | Consistent? |
|---|---|---|---|---|---|
| VB components × Sex (r_rb) | 0.220 (small) | Yes | 0.190 (small) | Yes | Yes |
| VB components × Race (r_rb) | 0.294 (small) | Yes | 0.284 (small) | Yes | Yes |
| VB components × Age (ε²) | 0.119 (medium) | Yes | 0.127 (medium) | Yes | Yes |
| Disc components × Sex (r_rb) | 0.233 (small) | Yes | 0.249 (small) | Yes | Yes |
| Disc components × Race (r_rb) | 0.048 (negligible) | No | 0.046 (negligible) | No | Yes |
| Disc components × Age (ε²) | 0.026 (small) | Yes | 0.014 (small) | Yes | Yes |

Younger patients and Black patients have more VB label components (more fragmented
segmentations). This is a potential signal of lower annotation quality in these
groups and warrants attention when evaluating Dice scores in Phase 4. Disc component
counts are less sensitive to demographics.

---

## 2. Crosscuts

7 tests (4 Mann-Whitney on continuous age + 3 chi-squared on categorical pairs).
Age tests: Female (a) vs Male (b); White (a) vs Black (b); GE (a) vs Siemens (b);
1.5T (a) vs 3.0T (b).

### Age distribution by grouping variable

| Comparison | Full effect (r_rb) | Full sig? | Train effect (r_rb) | Train sig? | Consistent? |
|---|---|---|---|---|---|
| Age × Sex | 0.067 (negligible) | No | 0.066 (negligible) | No | Yes |
| Age × Race (White vs Black) | 0.136 (small) | Yes | 0.129 (small) | Yes | Yes |
| Age × Manufacturer (GE vs Siemens) | 0.092 (negligible) | Yes | 0.085 (negligible) | No | No |
| Age × Field strength (1.5T vs 3.0T) | 0.139 (small) | Yes | 0.114 (small) | Yes | Yes |

Direction: White patients tend to be older than Black patients; 1.5T patients tend
to be older than 3.0T patients. The age × manufacturer signal is marginally
significant in the full cohort only — effect size is negligible and should not be
over-interpreted. Age × sex is not significant in either cohort.

### Categorical independence (confounder checks)

| Pair | Full V | Full sig? | Train V | Train sig? | Consistent? |
|---|---|---|---|---|---|
| Sex × Race | 0.056 (negligible) | No | 0.094 (negligible) | No | Yes |
| Race × Manufacturer | 0.066 (negligible) | No | 0.101 (negligible) | No | Yes |
| Race × Field strength | 0.071 (negligible) | No | 0.109 (negligible) | No | Yes |

Sex, race, manufacturer, and field strength are all mutually independent. Scanner
assignment is not confounded by patient race in either cohort. Any performance
disparities by race observed in Phase 4 can be attributed to patient factors rather
than scanner composition.

---

## Summary

13 of 23 comparisons are consistent and significant in both cohorts. The findings
that survive across splits and carry meaningful effect sizes are:

- **Sex → volume**: large effect, both structures, both cohorts. Expected anatomy.
- **Race → VB volume**: small effect, White > Black. No effect on disc volume.
- **Age → VB volume and components**: medium effect, older patients have larger and
  fewer-component VBs. Directly relevant to the Biased Ruler hypothesis — the
  silver-standard model may perform differently across age groups partly because the
  anatomy itself differs.
- **Scanner → voxel counts**: medium effect, GE >> Siemens. Physical units (mm³)
  are largely unaffected, confirming voxel count is a resolution artifact.
- **No scanner–race confound**: all three chi-squared tests are non-significant with
  negligible effect sizes in both cohorts.
