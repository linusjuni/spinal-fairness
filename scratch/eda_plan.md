# EDA Plan

To be done before the fairness study. Goal: understand distributions and spot issues that affect split design and subgroup selection.

## 1. Demographics

- Distribution of age, sex, race, ethnicity across the 1,232 patients.
- Identify groups too small for meaningful subgroup analysis (e.g. <20 patients).
- Check for missing or "Not Reported" values.

## 2. Scanner & Acquisition

- Distribution of manufacturer (GE vs Siemens), field strength (1.5T vs 3.0T).
- Slice thickness, pixel spacing, echo time, repetition time distributions.
- Flag any rare or unusual acquisition parameters.

## 3. Volume Properties

- Image dimensions (height, width, number of slices) per exam.
- Voxel spacing distributions.
- Identify outliers (unusually large/small volumes).

## 4. Segmentation Masks

- Per-volume voxel counts for each label (background, vertebral bodies, discs).
- Ratio of labeled vs background voxels.
- Flag outliers — volumes with suspiciously small or large segmentation regions.

## 5. Co-occurrence / Confounders

- Cross-tabulate demographics with scanner variables (e.g. race x field strength, age x manufacturer).
- Identify confounders: are certain demographics over-represented on certain scanners?
- This determines whether we can separate demographic effects from scanner effects in the fairness analysis.
