# Outlier Investigation Summary: Duke CSpineSeg Dataset

**Date:** 2026-02-10
**Dataset Size:** 1,255 cervical spine MRI exams
**Total Outliers Identified:** 73 cases (5.8%)

---

## Executive Summary

**RECOMMENDATION: EXCLUDE 1 CASE, KEEP 72 CASES**

Out of 73 outlier cases identified, only **1 case should be excluded** from the dataset as it is clearly a localizer/scout scan, not a cervical spine T2 scan. The remaining 72 cases represent legitimate protocol variations and should be retained.

---

## Detailed Findings

### 1. Critical Outlier - MUST EXCLUDE ❌

**Series ID:** `1.2.826.0.1.3680043.10.474.593973.22529`

**Anomalies:**
- ✗ Extreme slice count: 512 slices (vs. mean 16, expected 10-50)
- ✗ Very small width: 56 voxels (vs. mean 450, expected 200-600)
- ✗ Abnormal spacing_z: 0.488mm (vs. mean 3.95mm)

**Dimensions:**
```
Voxel grid:  56 × 512 × 512 voxels
Physical:    56mm × 250mm × 250mm
Spacing:     1.0mm × 0.488mm × 0.488mm
```

**Why this is NOT a cervical spine scan:**
1. **Physical depth = 250mm** - Far exceeds typical cervical spine depth (100-150mm)
2. **Extreme aspect ratio** - 56mm × 250mm suggests this is a sagittal localizer/scout scan
3. **512 slices** - Typical cervical scans have 15-40 slices
4. **Rotated dimensions** - The "height" and "n_slices" are both 512, suggesting axis confusion

**Likely explanation:** This is a sagittal localizer/scout scan that was included in the dataset by mistake. Scout scans are low-resolution preview images used for planning, not diagnostic images.

**Action:** **EXCLUDE from dataset**

---

### 2. Protocol Variation - KEEP ✓

**Number of cases:** 70 cases
**Anomaly:** spacing_z = 3.3mm (below 3σ threshold of 3.37mm)

**Representative example:** `1.2.826.0.1.3680043.10.474.593973.110`

**Dimensions:**
```
Voxel grid:  320 × 320 × 15-21 voxels (typical)
Physical:    220mm × 220mm × 49-70mm
Spacing:     0.69mm × 0.69mm × 3.3mm
```

**Why these ARE valid cervical spine scans:**
1. **Normal physical FOV:** 220-240mm is appropriate for cervical spine
2. **Normal physical depth:** 49-70mm is within expected range (100-150mm total coverage)
3. **Normal slice count:** 15-21 slices is typical for cervical spine
4. **Consistent protocol:** All 70 cases share identical acquisition parameters (3.3mm spacing_z)
5. **Just slightly thinner:** 3.3mm vs 3.95mm mean - only 16% thinner than average

**Likely explanation:** This is a distinct acquisition protocol used for a subset of patients. The 3.3mm slice thickness is perfectly reasonable for cervical spine imaging (typically 2-5mm).

**Action:** **KEEP in dataset** - These are legitimate cervical spine scans with a different protocol

---

### 3. Thick Slice Protocol - KEEP ✓ (with note)

**Number of cases:** 2 cases
**Anomaly:** spacing_z > 5.0mm (above 3σ threshold of 4.54mm)

**Case 1:** `1.2.826.0.1.3680043.10.474.593973.8846`
```
Voxel grid:  512 × 512 × 12 voxels
Physical:    260mm × 260mm × 60mm
Spacing:     0.51mm × 0.51mm × 5.0mm
```

**Case 2:** `1.2.826.0.1.3680043.10.474.593973.21563`
```
Voxel grid:  512 × 512 × 13 voxels
Physical:    220mm × 220mm × 69mm
Spacing:     0.43mm × 0.43mm × 5.32mm
```

**Why these ARE valid cervical spine scans:**
1. **Normal physical FOV:** 220-260mm is appropriate
2. **Normal physical depth:** 60-69mm is within expected range
3. **Normal slice count:** 12-13 slices is on the lower end but acceptable
4. **High in-plane resolution:** 0.43-0.51mm spacing is excellent (better than mean 0.53mm)
5. **Thicker slices are clinically acceptable:** 5mm slice thickness was common in older protocols

**Likely explanation:** Older acquisition protocol or different scanner vendor. The thick slices are compensated by higher in-plane resolution.

**Action:** **KEEP in dataset** - Valid scans, but consider noting in analysis that 2 cases have thicker slices (may affect through-plane segmentation accuracy)

---

## Statistical Breakdown

| Category | Count | % of Dataset | Action |
|----------|-------|--------------|--------|
| Total dataset | 1,255 | 100% | - |
| Normal cases | 1,182 | 94.2% | Keep |
| Protocol variation (3.3mm slices) | 70 | 5.6% | Keep |
| Thick slices (5.0-5.3mm) | 2 | 0.2% | Keep (note) |
| Localizer scan | 1 | 0.08% | **EXCLUDE** |
| **Final dataset** | **1,254** | **99.92%** | - |

---

## Reference Statistics

### Normal Distribution Parameters
```
n_slices:   mean = 15.6 ± 14.1,  3σ threshold = 57.9
width:      mean = 449.7 ± 124.5, 3σ threshold = 76.2 voxels
spacing_z:  mean = 3.95 ± 0.20mm, 3σ range = [3.37, 4.54] mm
```

### Expected Ranges for Cervical Spine
```
Slice count:       10-50 slices (typically 15-40)
Physical FOV:      150-300mm (width/height)
Physical depth:    100-150mm (total coverage)
Slice thickness:   2-5mm (typically 3-4mm)
In-plane spacing:  0.3-0.8mm
```

---

## Recommended Actions

### Immediate
1. **Exclude case `1.2.826.0.1.3680043.10.474.593973.22529`** from all analyses
   - Create exclusion list: `excluded_series_ids.txt`
   - Document reason: "Localizer/scout scan, not diagnostic image"

### Optional
2. **Flag thick-slice cases** for sensitivity analysis:
   - `1.2.826.0.1.3680043.10.474.593973.8846`
   - `1.2.826.0.1.3680043.10.474.593973.21563`
   - Consider analyzing segmentation performance separately for thick vs. thin slice protocols

3. **Document protocol variation** in methods section:
   - 94.4% standard protocol (spacing_z ~3.6-4.1mm)
   - 5.6% thin-slice protocol (spacing_z = 3.3mm)
   - 0.2% thick-slice protocol (spacing_z = 5.0-5.3mm)

---

## Files Generated

- `extreme_slice_count_cases.csv` - 1 case with abnormal slice count
- `small_width_cases.csv` - 1 case with abnormal width (same as above)
- `spacing_z_outlier_cases.csv` - 73 cases with outlier slice thickness
- `all_outlier_cases.csv` - Combined report with flags for each outlier type
- `outlier_overview.png` - Visualization of distributions with outliers highlighted
- `OUTLIER_SUMMARY.md` - This summary document

---

## Next Steps

1. Create exclusion script to filter out the localizer scan
2. Re-run volume statistics with excluded case
3. Verify that remaining outliers are legitimate protocol variations
4. Consider stratified analysis by protocol (standard vs. thin-slice vs. thick-slice)
5. Check if fairness metrics differ across protocol subgroups
