# Documentation Index

## Statistical Analysis

| Document | Contents |
|---|---|
| [Statistical Testing](statistical_testing.md) | Test battery for EDA (Mann-Whitney, Kruskal-Wallis, chi-squared), effect sizes, multiple testing correction, and when each technique is/isn't useful |

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
