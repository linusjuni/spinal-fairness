# Data Distribution & Confounders

**Dataset:** Duke CSpineSeg (after exclusions: N=1,254 exams, 1,231 patients)
**Date:** 2026-02-17

---

## 1. Demographic Distributions

### Sex (exam-level)
| Group | Count | % |
|-------|-------|---|
| Female | 684 | 54% |
| Male | 571 | 46% |

Slight female skew, but reasonably balanced for subgroup analysis.

### Race (exam-level)
| Group | Count | % | Viable for subgroup analysis? |
|-------|-------|---|------|
| White | 810 | 65% | Yes |
| Black or African American | 349 | 28% | Yes |
| Not Reported | 33 | 3% | No — exclude or merge |
| Other | 28 | 2% | No |
| Asian | 25 | 2% | No — too small |
| American Indian or Alaska Native | 9 | <1% | No |
| Native Hawaiian or Other Pacific Islander | 1 | <1% | No |

Heavily White-dominant. Only White and Black groups have sufficient sample sizes for reliable subgroup fairness analysis. Asian/AIAN/NHPI are too small. For the fairness study, the primary racial comparison will be **White vs. Black**, with remaining groups either excluded or reported descriptively without statistical testing.

### Ethnicity (exam-level)
| Group | Count | % |
|-------|-------|---|
| Not Hispanic or Latino | ~1,160 | 92% |
| Hispanic or Latino | ~63 | 5% |
| Not Reported | ~35 | 3% |

Hispanic group is very small (~63 exams). Statistical power for ethnicity-based fairness testing will be limited. Consider reporting descriptively only.

### Age
- Mean: ~55 years, SD: ~17
- Range: 18–90+
- Distribution: roughly normal with slight right skew, peak ~55–58
- Small secondary bump around age 25

---

## 2. Scanner Distributions

### Manufacturer
| Manufacturer | Count | % |
|---|---|---|
| Siemens | 789 | 63% |
| GE Medical Systems | 466 | 37% |

### Magnetic Field Strength
| Strength | Count | % |
|---|---|---|
| 1.5T | ~747 | 60% |
| 3.0T | ~508 | 40% |

### Acquisition Protocol Variations
| Protocol | Slice Thickness | Count | % |
|---|---|---|---|
| Standard | ~3.6–4.1mm | 1,182 | 94.2% |
| Thin-slice variant | 3.3mm | 70 | 5.6% |
| Thick-slice variant | 5.0–5.3mm | 2 | 0.2% |

### Volume Properties
- In-plane spacing: mean 0.53mm (range ~0.3–0.9mm)
- Slice thickness (spacing_z): mean 3.95mm, SD 0.20mm
- Grid sizes: mostly 320×320 or 512×512, with a few 256×256 and 768×768
- Slice count: mean 15.6, typically 12–21

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
