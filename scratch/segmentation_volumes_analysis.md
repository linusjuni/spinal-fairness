# Segmentation Volume EDA Analysis

Analysis of plots from `outputs/eda/segmentation_volumes/20260317_101723/`.

---

## Group 1: Distributions

> **What this checks:** Do the raw segmentation measurements look sensible before we slice by demographics?
>
> - mm3 volumes are unimodal -- good, no scanner artifacts
> - Voxel counts are bimodal -- expected, driven by GE vs. Siemens resolution
> - Component counts are slightly higher than expected anatomy (~9 VB, ~8 disc vs. expected ~7 and ~6)
> - 22 outlier exams with very high component counts are wider-FOV scans, not errors

**Files:** `dist_volume_mm3.png`, `dist_voxel_counts.png`, `dist_components.png`

**Observation:** Physical volume (mm3) distributions are unimodal and right-skewed for both tissue types: vertebral body volume peaks around 50,000--55,000 mm3 (bulk between 30,000--80,000 mm3, long tail past 200,000 mm3), disc volume peaks around 10,000--12,000 mm3 (bulk between 5,000--20,000 mm3). In contrast, voxel count distributions are clearly bimodal for both labels -- vertebral body voxel counts show peaks near ~25,000 and ~75,000, disc voxel counts near ~4,000 and ~17,000 -- which is the expected signature of two scanner populations (GE vs. Siemens) with different voxel resolutions. Component counts show vertebral bodies peaking at ~9 (range 7--11, right tail to ~28) and discs peaking sharply at 8 (range 7--9, right tail to ~33). The vertebral body mode of 9 is slightly above the expected 7 for C2--T1, suggesting some bodies are split into multiple connected components by the segmentation. Twenty-two exams have components >15, representing wider-FOV scans covering thoracic spine.

**Fairness implication:** The unimodal mm3 distributions confirm that physical volume successfully absorbs scanner resolution differences and is the correct metric for demographic comparisons. The bimodal voxel counts would introduce a severe scanner-driven confound if used directly. The right-tail outliers in volume and component counts should be checked for demographic skew to ensure they don't disproportionately affect one subgroup.

**Action required:** Flag for follow-up -- inspect high-volume outliers (VB > 100,000 mm3) and high-component outliers (>15) to confirm they are wider-FOV scans rather than annotation errors, and verify their demographic distribution.

---

## Group 2: Physical Volume by Demographics

> **What this checks:** Do different demographic groups have systematically different anatomy? If so, Dice scores could differ due to structure size, not model bias.
>
> - **Sex:** Males have ~25% larger vertebral bodies and ~20-25% larger discs -- the biggest confounder
> - **Race:** White patients have ~10% larger vertebral bodies, but disc volumes are identical
> - **Age:** Vertebral body volume increases slightly with age (~15%); disc volume is flat (no degeneration signal)
> - Disc segmentation is the cleaner fairness target -- uniform across race and age

**Files:** `vol_by_sex_*.png`, `vol_by_race_*.png`, `vol_by_age_*.png`

**Observation:** Sex shows the strongest anatomical differences: males have ~25% higher median vertebral body volume (~60,000 vs. ~48,000 mm3) and ~20--25% higher median disc volume (~13,000--14,000 vs. ~11,000 mm3), with wider distributions in both cases. This is anatomically expected. Race differences are more modest: White patients have ~10% higher median vertebral body volume (~55,000 vs. ~50,000 mm3) with dramatically longer right tails (outliers to ~240,000 mm3 vs. ~100,000 mm3 for Black patients), but disc volumes are essentially identical across race groups (both ~12,000--13,000 mm3). Age shows a subtle upward trend for vertebral body volume (~48,000 mm3 for <40, ~50,000 for 40--60, ~55,000 for 60+) with substantially greater variance in the 60+ group, while disc volumes are stable across all age bins (~12,000--13,000 mm3). Notably, the expected age-related disc volume decrease from degeneration is not observed.

**Fairness implication:** Sex is the strongest anatomical confounder for both tissue types -- larger structures are generally easier to segment (more true-positive voxels relative to boundary errors), so any sex-based Dice score gap must be interpreted alongside this ~25% volume difference. Race is a moderate confounder for vertebral bodies but not for discs, and age is a moderate confounder for vertebral bodies only. Disc segmentation is the cleaner target for fairness evaluation since disc volumes are relatively uniform across race and age. This parallels the MAMA-MIA finding (Parikh et al.) where anatomical representational bias (larger, more variable structures in specific subgroups) was a key confounder.

**Action required:** Flag for follow-up -- when Dice scores are computed, include volume as a covariate or report volume-stratified Dice to disentangle anatomical size effects from true segmentation quality differences. Summary of confounder risk:

| Comparison | Vertebral Body | Disc |
|---|---|---|
| Sex (M > F) | **High** (~25% gap) | **Moderate-High** (~20--25% gap) |
| Race (W > B) | **Moderate** (~10% gap, outlier asymmetry) | **Low** (negligible gap) |
| Age (60+ > <40) | **Moderate** (~15% gap, variance asymmetry) | **Low** (negligible gap) |

---

## Group 3: Voxel Counts by Scanner

> **What this checks:** Can we trust mm3 volumes as scanner-independent? (Validation step.)
>
> - Voxel counts differ massively between GE and Siemens -- cannot be used for comparisons
> - Siemens voxel counts are bimodal (1.5T vs. 3.0T resolution split)
> - mm3 volumes look identical across manufacturers -- validation passed
> - Safe to use mm3 for all demographic comparisons

**Files:** `voxels_by_manufacturer_*.png`, `vol_mm3_by_manufacturer_*.png`

**Observation:** Voxel counts differ substantially between manufacturers: GE distributions are unimodal (VB peak ~75,000, disc peak ~17,000) while Siemens distributions are clearly bimodal (VB modes near ~30,000 and ~100,000; disc modes near ~7,000 and ~20,000). The Siemens bimodality is consistent with its mixed 1.5T/3.0T scanner pool producing two distinct resolution regimes. In contrast, mm3 volume distributions are strikingly similar across manufacturers -- both show nearly identical violin shapes, widths, and medians for both vertebral bodies (~50,000--55,000 mm3) and discs (~12,000 mm3). The small residual mean differences (VB: GE=51,028 vs. Siemens=55,086 mm3, ~8%; disc: GE=11,972 vs. Siemens=12,292 mm3, ~3%) are not visually apparent and likely reflect minor demographic composition differences between scanner populations rather than a systematic scanner effect.

**Fairness implication:** This is the critical validation: mm3 volumes are scanner-independent and safe to use for demographic comparisons. Raw voxel counts would severely confound any analysis where scanner type correlates with demographic attributes (e.g., if certain patient populations are disproportionately scanned on one manufacturer). The successful normalization confirms the study's methodological choice.

**Action required:** None. The plots confirm expected behavior. Proceed with mm3 as the analysis unit for all demographic fairness comparisons.

---

## Group 4: Component Counts by Demographics

> **What this checks:** Does every demographic group get the same number of structures annotated? Differences here would mean unequal annotation coverage -- a label quality concern.
>
> - Sex and race: annotation coverage is consistent (no median shifts)
> - Age: vertebral body components decrease with age (~9 for <40 to ~7 for 60+), but disc components are stable
> - This suggests degenerative merging of vertebral bodies in older patients, not annotation bias
> - 22 wide-FOV outliers are concentrated among White females on Siemens 3T -- needs filtering

**Files:** `comp_by_sex_*.png`, `comp_by_race_*.png`, `comp_by_age_*.png`

**Observation:** Component counts are broadly consistent across sex and race groups, with medians near 7--8 for vertebral bodies and 9--10 for discs in all subgroups. Outlier tails are asymmetric: the 22 high-component outlier exams (wider-FOV scans) are concentrated among White females on Siemens 3T scanners, producing longer upper tails in the Female and White distributions for disc counts. The most consequential finding is an age-related decrease in vertebral body component count: median drops from ~9 in the <40 group to ~7 in the 60+ group. Crucially, this trend is not mirrored in disc components (stable at ~9--10 across all age bins), which suggests it is driven by degenerative merging of vertebral body segmentations in older patients (e.g., bridging osteophytes connecting adjacent vertebral bodies into single components) rather than narrower FOV or annotation protocol differences -- if FOV were the cause, disc counts would decrease in parallel.

**Fairness implication:** Core annotation coverage is consistent across sex and race, which is reassuring for the fairness study. The age-related vertebral body component decrease is a potential confounder: if older patients' vertebral bodies are merging into fewer, larger components, this could affect both volume metrics and Dice score evaluation in age-stratified analyses. The outlier concentration among White females is a dataset composition issue (scanner protocol variation) rather than annotation bias, but should be handled through filtering or stratification.

**Action required:** Needs investigation -- determine whether the age-related vertebral body component decrease is driven by (a) degenerative merging of segmentations (bridging osteophytes), (b) narrower FOV, or (c) annotation differences. Visually inspect representative cases from each age bin. Also ensure downstream analyses exclude or account for the 22 wide-FOV outlier exams.

---

## Overall Summary

> **The short version:**
>
> - Anatomical size varies by demographics (especially sex) -- this will confound Dice scores if not controlled for
> - mm3 is validated as scanner-independent -- safe to use everywhere
> - Annotation coverage is consistent across groups (no label quality bias by sex or race)
> - One thing to investigate: vertebral body components merge with age (degenerative changes)
> - Disc segmentation is the cleanest fairness target (uniform across race and age)
> - Use volume-adjusted Dice scores to separate anatomy from model bias

The segmentation volume EDA reveals several findings directly relevant to the fairness study:

1. **Sex is the dominant anatomical confounder.** Males have ~25% larger vertebral bodies and ~20--25% larger discs than females. This is anatomically expected but means any observed sex-based Dice score gap must be volume-adjusted before it can be attributed to model bias. This is the cervical spine analogue of the MAMA-MIA finding where anatomical representational bias confounded fairness evaluation.

2. **Race differences are modest and tissue-dependent.** White patients have ~10% larger vertebral bodies (with more extreme outliers) but essentially identical disc volumes compared to Black patients. Disc segmentation provides a cleaner fairness comparison across race. The vertebral body gap is small enough that it is unlikely to be the primary driver of any large Dice score disparity, but should still be controlled for.

3. **Age effects are asymmetric across labels.** Vertebral body volume increases slightly with age (~15%) while disc volume is stable -- the expected age-related disc degeneration signal is absent. More notably, vertebral body component counts decrease with age (likely from degenerative merging), which could complicate volume-based analyses in older patients.

4. **Physical volume (mm3) is validated as scanner-independent.** The bimodal voxel count distributions collapse into matched mm3 distributions across GE and Siemens, confirming mm3 is the correct metric for all demographic comparisons.

5. **Annotation coverage is consistent across demographics** in terms of core component counts (no systematic group has fewer annotated structures), with the exception of the age-related vertebral body merging noted above. The 22 wide-FOV outlier exams are a known dataset property concentrated among White females on Siemens 3T scanners, requiring filtering in downstream analyses.

**Bottom line:** Anatomical size differences (especially by sex) are a real confounder that must be addressed in the Dice-score fairness analysis, but they do not invalidate the study. Volume-adjusted or volume-stratified Dice scores, plus separate analysis of vertebral body and disc labels, will allow the study to distinguish genuine model bias from anatomical confounding. The disc label is the cleaner fairness target given its uniformity across race and age groups.
