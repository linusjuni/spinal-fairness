# Findings — MRI-CORE MVP Probe

Date: 2026-04-21. Encoder: MRI-CORE ViT-B (768-d, mid-sagittal slice).
Pipeline: frozen features → PCA(50) → logistic regression, 5-fold CV.
CIs are Nadeau-Bengio corrected.

## Linear probe results

| Attribute | Metric | Mean | ±95% CI (NB) |
|---|---|---|---|
| Sex | AUROC | **0.931** | ±0.031 |
| Age (3-bin) | Balanced accuracy | **0.632** | ±0.043 |
| Race (3-group) | Balanced accuracy | 0.513 | ±0.071 |
| Manufacturer | AUROC | **0.997** | ±0.007 |

Random baseline for 3-class balanced accuracy is 0.333.

## Interpretation

**Sex (0.931)** — well above the 0.80 strong-leakage threshold. MRI-CORE
strongly encodes biological sex. PCA(2) scatter showed no visible separation,
confirming the signal lives beyond the top 2 components.

**Age (0.632)** — moderate signal above chance. Below the 0.80 threshold
but meaningfully above random.

**Race (0.513)** — weak. With the NB CI of ±0.071 the upper bound is 0.584
— below the 0.65 "investigate confounds" threshold. No strong evidence of
race encoding in MRI-CORE features.

**Manufacturer (0.997)** — near-perfect scanner fingerprinting, as expected
for an encoder pretrained on Duke institutional data.

## Scanner confound check

Potential concern: if MRI-CORE recognises Siemens vs GE at 0.997 AUROC, and
sex correlated with scanner assignment, the sex signal could be artifactual.

EDA crosscut (`full/crosscuts`, 2026-04-21): sex × manufacturer chi2
p=0.210, Cramér's V=0.035 — **not significant**. Female: 61% Siemens
(418/683). Male: 65% Siemens (370/571). Near-identical proportions.

**The scanner confound does not explain the sex AUROC.** Males and females
are scanned on Siemens and GE in essentially equal proportions.

Race × manufacturer was also not significant (p=0.49, V=0.066), consistent
with prior EDA.

## Next steps

- Run non-Duke encoders (Triad-SwinB, MedicalNet) to confirm sex signal
  generalises beyond MRI-CORE's institutional pretraining.
- Add UMAP(2) and per-PC K–S tests (Glocker 2023 style).
- Permutation test on sex AUROC (N=1000 label shuffles).
- Age as continuous regression (R²) rather than 3-bin classification.
