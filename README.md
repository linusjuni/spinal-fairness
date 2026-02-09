# Cervical Spine Segmentation Fairness

> 🚧 **Work in Progress** — This project is in early development

Investigating algorithmic bias in automated medical image segmentation using the Duke CSpineSeg dataset.

## Overview

This project examines fairness in cervical spine MRI segmentation, focusing on how machine-generated labels ("Silver Standard") may introduce or amplify bias compared to expert annotations ("Gold Standard") across age and demographic groups.

## Key Objectives

- **Fairness Audit**: Quantify performance disparities across patient subgroups
- **Label Bias Analysis**: Compare expert vs. automated annotation quality
- **Label Classification**: Develop methods to distinguish annotation sources

## Methods

- **Dataset**: Duke CSpineSeg (vertebral bodies & intervertebral discs)
- **Architecture**: 3D nnU-Net baseline with potential transformer variants
- **Infrastructure**: MLOps pipeline for reproducible experiments

## References

Based on recent work examining bias in medical imaging datasets (MAMA-MIA) and the "Biased Ruler" phenomenon in automated labeling systems.

---
**Student**: Linus Juni

**Supervisors**: Aditya Parikh, Aasa Feragen

**Institution**: DTU Compute
