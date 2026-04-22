# Special Course Project Plan

## Fairness in Cervical Spine MRI Segmentation
*Quantifying and Mitigating Bias in Automated Labels*

**Linus Juni**  
Student ID: 225224

**Supervised by**  
Aditya Parikh and Aasa Feragen

Department of Applied Mathematics and Computer Science  
January 8, 2026

---

## Contents

1. [Project Overview](#1-project-overview)
2. [Research Goals](#2-research-goals)
3. [Infrastructure and Reproducibility](#3-infrastructure-and-reproducibility)
4. [Methods](#4-methods)
5. [Risk Analysis](#5-risk-analysis)

---

## 1. Project Overview

We will explore algorithmic fairness in medical image segmentation using the Duke University Cervical Spine MRI (CSpineSeg) dataset. Our primary interest is the "Biased Ruler" effect, where machine-generated labels used for validation can actually hide or exaggerate how well a model performs across different patient subgroups. To allow for reproducibility of all experiments, we will implement an MLOps-inspired framework focused on consistent results and auditability following best ML practices. This infrastructure will help us track experiments reliably and make sure the entire pipeline can be easily monitored and verified.

## 2. Research Goals

- **Fairness Audit**: Measure how segmentation performance varies across different patient ages and demographics within the CSpineSeg cohort.

- **Label Bias Investigation**: Compare expert-annotated "Gold-Standard" labels against automated "Silver-Standard" labels to see if machine-generated annotations carry systematic flaws.

- **Bias Amplification**: Investigate the phenomenon of bias amplification by testing whether training on machine-generated labels widens performance gaps in the spine context, as seen in recent literature.

- **Correction Strategies**: Explore how a small amount of expert guidance or semi-supervised learning can help fix these fairness gaps.

- **Label Classification**: Develop a computer vision – most likely deep learning – approach to distinguish between expert-annotated and machine-generated labels.

- **Metric Standardization**: Define and validate a suite of fairness metrics specifically tailored for medical segmentation tasks.

- **XAI Exploration (Optional)**: Use explainability methods to identify the qualitative features or image characteristics that cause the model to fail in certain groups.

## 3. Infrastructure and Reproducibility

Rather than just monitoring metrics, our architecture will focus on creating a publication-ready workflow:

- **Experiment Tracking**: Log every training run to ensure that results can be exactly reproduced and verified later.

- **Scalable Pipeline**: Set up a robust data handling system for 3D MRI volumes to ensure the model training process is consistent.

- **Code and Data Integrity**: Implement versioning for both the model code and the specific data splits used during the fairness audit.

## 4. Methods

- **Data**: Use the CSpineSeg dataset, focusing on the segmentation of vertebral bodies and intervertebral discs.

- **Model**: Implement 3D segmentation models using the nnU-Net framework, which has shown strong baseline performance in spine imaging – potentially exploring other architectures such as Transformers.

- **Evaluation**: Assess the model's fairness and performance by comparing metrics across demographic subgroups, allowing for the refinement of specific evaluation strategies as the research progresses.

## 5. Risk Analysis

- **Computational Resources**: High demand for GPU memory during 3D model training.

- **Time Management**: Ensuring the MLOps setup does not detract from the core fairness analysis deadline.
