# Documentation Index

## Project Plan

| Document | Contents |
|---|---|
| [Project Plan](project-plan/project-plan.md) | Formal course project plan — research goals, infrastructure, methods, risks |

## Exploratory Data Analysis

| Document | Contents |
|---|---|
| [Data Distribution](eda/data-distribution.md) | Demographic and scanner counts, cross-demographic confounder analysis, implications for study design |
| [Segmentation Volumes](eda/segmentation-volumes.md) | Plot-by-plot analysis of segmentation volumes (mm³ vs voxels, by sex/race/age/scanner) |
| [Comparison: Full vs Train](eda/comparison.md) | Statistical test results (effect sizes, BH-FDR) comparing full cohort and train split |

## Methodology

| Document | Contents |
|---|---|
| [Statistical Testing](statistical-testing/statistical-testing.md) | Test battery for EDA (Mann-Whitney, Kruskal-Wallis, chi-squared), effect sizes, multiple testing correction, and when each technique is/isn't useful |
| [Splits](splits/splits.md) | Sex-balancing decision — split v1/v2/v3 explored, v3 (exam-level 50/50 downsample) adopted |

## nnU-Net Pipeline

| Document | Contents |
|---|---|
| [01 — Setup](nnunet/01_setup.md) | Environment variables, directory layout, known dependency issues |
| [02 — Dataset Preparation](nnunet/02_dataset_preparation.md) | nnU-Net format, naming, dataset.json, demographic splits |
| [03 — Training](nnunet/03_training.md) | Preprocessing, job scripts, custom trainer, submitting jobs |
| [04 — Inference & Evaluation](nnunet/04_inference.md) | Model selection, prediction, postprocessing, metrics, fairness analysis |

## Demographic Probing of Medical Image Encoders

Diagnostic pipeline that checks whether demographic signal is latently
encoded in MRI features — precursor to any debiasing work.

| Document | Contents |
|---|---|
| [Encoder Recommendations](demographic-probing-of-medical-image-encoders/encoder-recommendations.md) | MedicalNet / RadImageNet / SAM-Med3D survey, tradeoffs, gotchas |
| [Papers](demographic-probing-of-medical-image-encoders/papers.md) | Gichoya, Glocker, FairMedFM, supervisor prior work |
| [Methodology](demographic-probing-of-medical-image-encoders/methodology.md) | Preprocessing, PCA+UMAP, probe AUROC, permutation test, confounds |
| [Sketch](demographic-probing-of-medical-image-encoders/sketch.md) | Flow diagram, MVP plan, output structure |
| [Findings](demographic-probing-of-medical-image-encoders/findings.md) | MRI-CORE MVP probe results (2026-04-21) |
