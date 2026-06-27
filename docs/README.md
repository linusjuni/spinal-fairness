# Documentation Index

This is the documentation root for the **spinal-fairness** project (fairness in
cervical-spine MRI segmentation). Use the folder map below to jump to a topic,
then the per-folder tables for individual documents.

## Folder Map

| Folder | What lives here |
|---|---|
| [`project-plan/`](#project-plan) | Formal course project plan — goals, methods, risks |
| [`faimi/`](#faimi-submission) | FAIMI workshop submission requirements (page limits, format, deadlines) |
| [`eda/`](#exploratory-data-analysis-eda) | Exploratory data analysis — distributions, volumes, full-vs-train comparison |
| [`statistical-testing/`](#methodology) | Statistical test battery used across EDA |
| [`splits/`](#methodology) | Train/val/test split design and sex-balancing decisions |
| [`nnunet/`](#nnu-net-pipeline) | nnU-Net pipeline — setup → preparation → training → inference → release |
| [`fairness/`](#fairness-analysis) | Fairness evaluation runs and outlier diagnostics |
| [`demographic-probing-of-medical-image-encoders/`](#demographic-probing-of-medical-image-encoders) | Probing whether demographic signal is latent in MRI encoders |
| [`meetings/`](#meeting-notes) | Supervisor meeting notes (by date) |
| [`papers/`](#reference-papers) | Reference PDFs (supervisor prior work + dataset paper) |

## Project Plan

| Document | Contents |
|---|---|
| [Project Plan](project-plan/project-plan.md) | Formal course project plan — research goals, infrastructure, methods, risks |

## FAIMI Submission

| Document | Contents |
|---|---|
| [Submission Requirements](faimi/submission-requirements.md) | FAIMI workshop hard constraints — 8-page limit (+2 for references), LNCS template, double-blind anonymization, CMT platform, supplementary-material rules, and dates (based on FAIMI 2024/2025; 2026 CfP TBA) |
| [Content Cut from paper/](faimi/content-cut-from-paper.md) | Per-section log of what was dropped/shortened when compressing the long `paper/` draft into the 8-page `submission/` — so cuts are deliberate and reinstatable (e.g. the connected-components degeneration finding, `tab:notation`, `tab:volumes`) |
| [Paper Framing & Positioning](faimi/paper-framing.md) | How to frame the paper for FAIMI — the biased-ruler lineage (Parikh et al., which we *extend*, not invent) reconciled with Aasa's validation-problem headline and Aditya's "first cervical-spine fairness audit" hook; our novelty as a complementary "false confidence" mode; revised contributions list, the ~8-point leakage angle, and the pseudo-random-assignment risk to back up |

## Exploratory Data Analysis (EDA)

| Document | Contents |
|---|---|
| [Data Distribution](eda/data-distribution.md) | Demographic distributions (sex, race, ethnicity, age), scanner distributions (manufacturer, field strength, protocols), and confounder analysis (race–scanner independence, age confounders) with implications for fairness study design |
| [Segmentation Volumes](eda/segmentation-volumes.md) | Segmentation volume distributions — physical volumes by sex/race/age, scanner voxel-count validation, component-count patterns across demographics, and fairness implications |
| [Comparison: Full vs Train](eda/comparison.md) | Statistical results comparing full cohort vs train split across volumes, demographic effects, scanner effects, annotation completeness, and confounder checks (effect sizes, significance) |

## Methodology

| Document | Contents |
|---|---|
| [Statistical Testing](statistical-testing/statistical-testing.md) | Test battery for EDA (Mann-Whitney, Kruskal-Wallis, chi-squared), effect sizes, multiple testing correction, and when each technique is/isn't useful |
| [Splits](splits/splits.md) | Sex-balancing decision — splits v1/v2/v3 explored, v3 (exam-level 50/50 downsample) adopted |

## nnU-Net Pipeline

| Document | Contents |
|---|---|
| [01 — Setup](nnunet/01_setup.md) | PyTorch/nnunetv2 installation, required environment variables (raw/preprocessed/results dirs), and CSpineSeg directory layout |
| [02 — Dataset Preparation](nnunet/02_dataset_preparation.md) | nnU-Net format for Dataset001_CSpineSeg — file naming, dataset.json (channels, labels), train/val/test layout, and stratified 5-fold CV setup |
| [03 — Training](nnunet/03_training.md) | Preprocessing, write_splits, 10 training jobs (2d + 3d_fullres), custom `nnUNetTrainerWandB` with W&B logging, job submission/monitoring/resume, and output structure |
| [04 — Inference & Evaluation](nnunet/04_inference.md) | Five-step pipeline — find best configuration, predict on test set, postprocess, compute metrics (Dice, IoU), and fairness analysis across sex/race/age |
| [05 — Model Selection & Test Evaluation](nnunet/05_model_selection.md) | Training status, ensemble Dice vs mixed/gold/silver references, per-fold CV Dice, comparison with Zhou et al. baseline, and per-case macro Dice distribution |
| [06 — Gold / Silver Label Experiment](nnunet/06_gold_silver_training.md) | Biased-ruler experiment — Dataset001 (mixed), Dataset002 (gold), Dataset003 (silver); pipeline diagram, MAMA-MIA contrast, per-dataset train/predict/eval steps and results; bias amplification complete (Run 9, no amplification found) |
| [07 — Hugging Face Release Plan](nnunet/07_huggingface_release.md) | Export models to zip, upload to private HF repo, and write a model card (CC-BY-NC-4.0 license, input specs, performance tables, fairness caveats, inference instructions) |
| [08 — Carbon Footprint of Training](nnunet/08_carbon_footprint.md) | GPU-hours by hardware (A100/L40S), ML-CO₂ emissions estimate (~50–70 kg CO₂e), methodology, scenario/sensitivity table, suggested paper wording, and BibTeX citations |

## Fairness Analysis

| Document | Contents |
|---|---|
| [DPD/DIR Redefinition](fairness/dpd-dir-redefinition.md) | **Migration note (2026-06-07)** — DPD/DIR changed from continuous mean-ratio to the canonical binarized rate-based definition (Parikh et al. / Fairlearn / four-fifths). Read this first: it explains why older mean-based numbers are superseded and what the binarized reruns show (incl. silver-ruler saturation) |
| [Fairness Runs](fairness/fairness-runs.md) | Chronological catalog of fairness evaluation runs (1–9) — rulers, metrics, test cases, and Dice/HD95/nDSC findings across 7 demographic groupings. Runs 1–6 are the old mean-based definition (superseded); Runs 7–8 are the binarized reruns; Run 9 is the bias amplification result (DS001/DS002/DS003, no amplification found) |
| [Biased Ruler — Silver-Source Limitation & Options](fairness/biased-ruler-silver-source.md) | **Decision note (2026-06-09).** We reproduce Parikh et al.'s verdict-flip (same model age-fair vs gold 0/63, age-unfair vs silver 11/63) via a *complementary* mechanism — our correlated-twin silver manufactures *false confidence* (variance collapse), not *false magnitude*; the age effect is intrinsic (survives both rulers + the encoder probe). Compares our findings to Parikh et al. (direction flipped, magnitude tiny) and weighs the options (finish amplification vs build a strongly-biased independent silver) — a supervisor-meeting item |
| [HD95 Outliers](fairness/hd95-outliers.md) | **Superseded (2026-06-07).** The old mean-vs-median HD95 DIR workaround — no longer needed now that binarized HD95 DIR is outlier-robust by construction. Retained for provenance |
| [Related Work & Novelty](fairness/related-work.md) | Positioning vs prior fairness literature — FairMedFM (the one prior spine-segmentation fairness work, sex-only/lumbar) and how our scope (cervical, sex+race+age, biased ruler) is novel; how to phrase the "first" claim |

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

## Meeting Notes

Supervisor meetings (Linus Juni with Aditya Parikh). Filenames are dates in
DD-MM-YYYY format.

| Document | Date | Contents |
|---|---|---|
| [11-03-2026](meetings/11-03-2026) | 11 Mar 2026 | Data splits with sex balancing, formal statistical-testing requirements, EDA consolidation + deep volumetric analysis, and automated segmentation quality checks |
| [11-04-2026](meetings/11-04-2026) | 11 Apr 2026 | Completion of sex-balanced exam-level splits (v3), statistical testing for group comparisons, structured EDA tables, and segmentation quality-assessment libraries |
| [14-04-2026](meetings/14-04-2026) | 14 Apr 2026 | Baseline fairness audit across strata, class-conditional training for mitigation, confident learning for label noise, statistical tests, vertebral morphology, and representation-space debiasing |
| [01-05-2026](meetings/01-05-2026) | 1 May 2026 | Biased-ruler methodology (global predictions as silver labels), need for gold-only/silver-only models, EDA by annotation quality, nnU-Net encoder probing, and exam/paper milestones |

## Reference Papers

| Document | Title & Authors | Contents |
|---|---|---|
| [papers/aditya1.pdf](papers/aditya1.pdf) | *Investigating Label Bias and Representational Sources of Age-Related Disparities in Medical Segmentation* — Parikh, Das, Feragen (DTU Compute) | Algorithmic bias in breast-cancer MRI segmentation — systematic disparities against younger patients from label quality and representational imbalance in MAMA-MIA |
| [papers/aditya2.pdf](papers/aditya2.pdf) | *Who Does Your Algorithm Fail? Investigating Age and Ethnic Bias in the MAMA-MIA Dataset* — Parikh, Das, Feragen (DTU Compute) | Age/ethnic bias in automated breast-cancer segmentation labels — intrinsic age bias persists after controlling for confounders; label-quality vs representation sources |
| [papers/spine.pdf](papers/spine.pdf) | *The Duke University Cervical Spine MRI Segmentation Dataset (CSpineSeg)* — Zhou, Wiggins, Zhang, Colglazier, Willhite, Dixon, Malinzak, Gu, Mazurowski, Calabrese | The CSpineSeg dataset — 1,255 sagittal T2-weighted MRI scans from 1,232 patients with expert vertebral-body and intervertebral-disc segmentations |
