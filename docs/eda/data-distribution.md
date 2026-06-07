# Data Distribution & Confounders

**Dataset:** Duke CSpineSeg (after exclusions: N=1,254 exams, 1,231 patients)
**Date:** 2026-02-17 | Cohort flow verified 2026-06-05 | Distributions re-verified
2026-06-07 against the 1,254 working set via `scripts/cohort_composition.py`

> **Note:** the §1–§2 distribution counts below are on the **1,254 working set**
> (post-exclusion). An earlier revision reported these on the 1,255 public release,
> which inflated White / Female / Siemens / 1.5T by one each — the single excluded
> localizer scan was a White female on a Siemens 1.5T scanner.

---

## 0. Cohort Flow

### From source paper to study cohort

| Stage | Exams | Patients | Action | Source |
|-------|-------|----------|--------|--------|
| CSpineSeg public release | 1,255 | 1,232 | — | Zhou et al., *Sci. Data* 2025 |
| After exclusions (`exclusions.py`) | 1,254 | 1,231 | −1 localizer/scout scan (56×512×512 voxels; abnormal dimensions, wrong scan type) | `src/data/exclusions.py` |
| After v3 sex-balancing (split_v3) | **1,142** | — | −112 excess female exams randomly downsampled (80 train / 7 val / 25 test) to achieve exact 50/50 M/F in every split | `src/data/splits/v3.py` |

The 1,254 figure is the **study working set** — all cases with valid metadata. The 1,142 figure is the **analysis cohort** — what actually enters training, validation, and test sets. The gap of 112 is entirely attributable to the deliberate sex-balancing downsample, not data quality issues.

### Why sex-balance by downsampling?

The source cohort is ~55% female / ~45% male (683 F / 571 M in the 1,254 working set), reflecting the clinical population. For the fairness audit, controlled sex-balance in both training and evaluation sets is necessary to isolate sex-based performance gaps from imbalance effects. Following Aditya et al. (MAMA-MIA), the approach is to downsample the majority sex (female) rather than upsample. See `docs/splits/splits.md` for full split version history.

### Gold / silver label pool (in the 1,142 analysis cohort)

| Pool | Exams | Overlap with other pool |
|------|-------|------------------------|
| Gold (expert annotations) | 448 | 0 (mutually exclusive by design) |
| Silver (auto-generated) | 694 | 0 |

Every exam in the analysis cohort carries exactly one label type. Verified: `split_v3_gold.tsv` ∩ `split_v3_silver.tsv` = ∅.

---

## 1. Demographic Distributions

### Sex (exam-level)
| Group | Count | % |
|-------|-------|---|
| Female | 683 | 54.5% |
| Male | 571 | 45.5% |

Slight female skew, but reasonably balanced for subgroup analysis.

### Race (exam-level)
| Group | Count | % | Viable for subgroup analysis? |
|-------|-------|---|------|
| White | 809 | 64.5% | Yes |
| Black or African American | 349 | 27.8% | Yes |
| Not Reported | 33 | 2.6% | No — exclude or merge |
| Other | 28 | 2.2% | No |
| Asian | 25 | 2.0% | No — too small |
| American Indian or Alaska Native | 9 | 0.7% | No |
| Native Hawaiian or Other Pacific Islander | 1 | 0.1% | No |

Heavily White-dominant. Only White and Black groups have sufficient sample sizes for reliable subgroup fairness analysis. Asian/AIAN/NHPI are too small. For the fairness study, the primary racial comparison will be **White vs. Black**, with remaining groups either excluded or reported descriptively without statistical testing.

### Ethnicity (exam-level)
| Group | Count | % |
|-------|-------|---|
| Not Hispanic or Latino | 1,162 | 92.7% |
| Hispanic or Latino | 59 | 4.7% |
| Not Reported | 33 | 2.6% |

Hispanic group is very small (59 exams). Statistical power for ethnicity-based fairness testing will be limited. Consider reporting descriptively only.

### Age
- Mean: 54.6 years, SD: 16.3, median: 56
- Range: 18–89 (13 exams have missing age, stored null, confirmed >89)
- Distribution: roughly normal with slight right skew, peak ~55–58
- Small secondary bump around age 25

---

## 2. Scanner Distributions

### Manufacturer
| Manufacturer | Count | % |
|---|---|---|
| Siemens | 788 | 62.8% |
| GE Medical Systems | 466 | 37.2% |

### Magnetic Field Strength
| Strength | Count | % |
|---|---|---|
| 1.5T | 746 | 59.5% |
| 3.0T | 508 | 40.5% |

### Acquisition Protocol Variations
| Protocol | Slice Thickness | Count | % |
|---|---|---|---|
| Standard | ~3.6–4.1mm | 1,182 | 94.2% |
| Thin-slice variant | 3.3mm | 70 | 5.6% |
| Thick-slice variant | 5.0–5.3mm | 2 | 0.2% |

### Volume Properties
_Source: `outputs/eda/full/mri_volumes/.../stats.json` (n=1,254 working set)._
- In-plane spacing: mean 0.53mm, SD 0.15mm
- Slice thickness (spacing_z): mean 3.96mm, SD 0.17mm
- Grid sizes: mostly 320×320 or 512×512, with a few 256×256 and 768×768 (mean 450±124, min 256, max 768; in-plane matrix is square)
- Slice count: mean 15.2, SD 1.3, range 12–25 (right-skewed: 80 scans beyond 2σ, 0 beyond 3σ)

---

## 3. Confounder Analysis (Cross-Demographic)

### Race × Scanner — NOT confounded (good)
Race groups are distributed roughly proportionally across manufacturers and field strengths:
- White: 38% GE / 62% Siemens
- Black: 35% GE / 65% Siemens
- Similar proportional split for 1.5T vs 3.0T across race groups

This means scanner-related performance differences are unlikely to masquerade as racial fairness gaps. No adjustment needed.

### Age × Sex — NOT confounded
Nearly identical age distributions for males and females. No concern.

### Age × Manufacturer — NOT confounded
Similar violin plots for GE and Siemens. No concern.

### Age × Field Strength — NOT confounded
Similar distributions for 1.5T and 3.0T. No concern.

### Age × Race — POTENTIAL CONFOUNDER ⚠️
- White patients skew slightly older (median ~60)
- Black patients have a similar median but wider spread
- "Other" category skews younger

If segmentation performance varies with age (e.g., older patients have more degenerative changes, osteophytes, fused segments, disc collapse — all of which make segmentation harder), then age differences across racial groups could partially explain apparent racial disparities. **Must control for age in fairness analysis** (e.g., age-stratified comparisons or regression with age as covariate).

### Sex × Race — MILD
Black patients have a slightly higher female:male ratio (56:44) vs. White (54:46). Likely not significant but worth noting.

### Protocol × Demographics — NOT YET ASSESSED
The 70 thin-slice cases (3.3mm) and 2 thick-slice cases should be checked for demographic distribution. If protocol variation correlates with specific demographics, it becomes a confounder.

---

## 4. Implications for Study Design

### Viable Fairness Axes
1. **Sex** (Female vs. Male) — well-balanced, no major confounders
2. **Race** (White vs. Black) — sufficient samples, age is a confounder to control for
3. **Age** (binned into groups, e.g., <40 / 40–60 / 60+) — continuous variable, can also serve as confounder control

### Limited/Descriptive Only
4. **Ethnicity** (Hispanic vs. Non-Hispanic) — small Hispanic group (~63), report descriptively
5. **Minority races** (Asian, AIAN, NHPI) — too few samples for statistical testing

### Scanner as Confounder vs. Fairness Axis
- Manufacturer and field strength can be analyzed as their own fairness axes (does the model perform worse on GE vs. Siemens? 1.5T vs. 3.0T?)
- Since they're NOT confounded with demographics, they can also be included as covariates in demographic fairness analysis without collinearity concerns

### Key Statistical Controls
- **Always control for age** when comparing race groups
- **Consider controlling for scanner** (manufacturer, field strength) as a secondary check, even though cross-tabs suggest no confounding
- **Protocol variation** (slice thickness) should be tracked but the 70 thin-slice cases are a small fraction and unlikely to drive results
