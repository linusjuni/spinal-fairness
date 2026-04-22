# Findings — MRI-CORE MVP Probe

Pipeline: frozen features → PCA(50) → logistic regression, 5-fold CV.
CIs are Nadeau-Bengio corrected.

## Linear probe results

### MRI-CORE uncropped (2026-04-21)

Encoder: MRI-CORE ViT-B (768-d, mid-sagittal slice, no foreground crop).

| Attribute | Metric | Mean | ±95% CI (NB) |
|---|---|---|---|
| Sex | AUROC | **0.931** | ±0.031 |
| Age (3-bin) | Balanced accuracy | **0.632** | ±0.043 |
| Race (3-group) | Balanced accuracy | 0.513 | ±0.071 |
| Manufacturer | AUROC | **0.997** | ±0.007 |

Random baseline for 3-class balanced accuracy is 0.333.

### MRI-CORE cropped (2026-04-22)

Encoder: MRI-CORE ViT-B (768-d, mid-sagittal slice, bounding-box foreground crop
at 5% of max intensity before resize to 1024²).

| Attribute | Metric | Mean | ±95% CI (NB) | Δ vs uncropped |
|---|---|---|---|---|
| Sex | AUROC | **0.936** | ±0.024 | +0.005 |
| Age (3-bin) | Balanced accuracy | 0.623 | ±0.091 | −0.009 |
| Race (3-group) | Balanced accuracy | 0.515 | ±0.080 | +0.002 |
| Manufacturer | AUROC | **0.997** | ±0.008 | 0.000 |

All differences are within the confidence intervals — no attribute changed meaningfully
under foreground cropping.

## Interpretation

**Sex (~0.93)** — well above the 0.80 strong-leakage threshold in both variants.
MRI-CORE strongly encodes biological sex. PCA(2) scatter showed no visible separation,
confirming the signal lives beyond the top 2 components.

**Age (~0.63)** — moderate signal above chance. Below the 0.80 threshold
but meaningfully above random.

**Race (~0.51)** — weak. Upper bound (mean + CI) is below the 0.65 "investigate
confounds" threshold. No strong evidence of race encoding in MRI-CORE features.

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

## Caveats

1. **Body-extent confound — one sub-variant ruled out, not the confound in general.**
   "Body extent" is actually three nested sub-confounds: body-to-image *ratio* (how
   much of the frame is tissue), silhouette *shape* (the body contour itself), and
   background *intensity/noise* statistics. The cropped variant (`mri_core_cropped`)
   was a cheap test of the first one only. A bbox crop + resize to 1024² normalises
   the body-to-air fill ratio — after cropping, every image has tissue filling ~100 %
   of the frame instead of the variable padding seen in the uncropped cohort.

   Sex AUROC moved +0.005 (0.931 → 0.936). **Reading: body-to-air fill ratio is
   probably not the driver of the sex signal.**

   Two caveats on that null result, both of which temper how much weight it carries:

   - **The cropped output still contains substantial air.** Visual inspection of
     `outputs/probe/preprocessing_preview.png` showed cropped network inputs with
     large black regions around the body silhouette — the image looked nearly
     identical to the uncropped version. Bbox cropping only removes rows/columns
     that are *entirely* below the 5 %-of-max threshold, which is a loose
     constraint: anatomy is non-rectangular (tapering head / neck outline) so a
     tight bbox still encloses lots of air around curved edges, and any single
     above-threshold pixel in a row (nose tip, noise spike) keeps the full width
     of background around it. The fill-ratio normalisation this variant was
     supposed to deliver barely happened, which weakens the null AUROC result —
     a near-no-op intervention can't refute the confound it's nominally testing.
   - **Square resize introduces a new sex-linked stretch artefact.** Resizing
     rectangular bboxes (e.g. 600×800 vs 700×800) to 1024² square stretches the
     axes unequally, and aspect ratio of the cropped body silhouette is itself
     sex-linked (men tend to have wider necks per unit height). The cropped variant
     may have traded a fill-ratio confound for an anisotropic-stretch one. Aspect-
     preserving resize (letterbox-pad to 1024²) would be the cleaner input-level
     ablation if we ever revisit this route.

   Silhouette shape and background intensity/noise are both untouched by bbox
   cropping, and the silhouette signal is probably the larger concern anyway. Otsu
   + `binary_fill_holes` pixel masking was considered and dropped: it creates a
   hard silhouette edge that is out-of-distribution for MRI-CORE (pretrained on raw
   MRI with air) and does not touch the silhouette-shape signal, so it would
   introduce new artefacts without resolving the bigger one. The random-init null
   below is the cleaner next move — see "Next steps".

2. **Duke institutional pretraining.** MRI-CORE was pretrained on Duke
   clinical data (2016–2020), the same institution and scanner pool as
   CSpineSeg. Cervical anatomy was deliberately excluded from pretraining
   (no data-leakage issue), but the encoder may encode Duke-specific
   acquisition fingerprints (protocols, vendor mix, post-processing) that
   incidentally correlate with sex. Cross-encoder comparison with non-Duke
   encoders (Triad-SwinB, MedicalNet) is the planned disentanglement.

3. **MRI-CORE uses its own normalisation.** The project-standard clip +
   z-score preprocessing (methodology.md) is deliberately bypassed for
   MRI-CORE — pretraining expects per-slice min-max + ImageNet mean/std,
   and feeding differently-normalised inputs breaks the features. Other
   encoders in the lineup will use different preprocessing.

## Preprocessing — current status

| Step | Status | Notes |
|---|---|---|
| RAS reorient | done | `nib.as_closest_canonical` |
| Mid-sagittal slice | done | `vol[X // 2]` after RAS |
| Per-slice min-max to [0, 1] | done | Matches MRI-CORE pretraining |
| Grayscale → 3-channel | done | SAM/DINOv2 expects RGB |
| Resize to 1024² (bilinear) | done | SAM input size |
| ImageNet mean/std | done | Matches MRI-CORE pretraining |
| Foreground crop (bbox @ 5% max) | done | Ran 2026-04-22; no meaningful effect on probe AUROCs. See Caveat 1 — the crop was near-no-op in practice (cropped output still shows substantial air), so this is not a strong refutation of the body-extent confound. |
| In-plane resample to fixed grid | not done | Min-max + resize approximates it; revisit if anatomy scale becomes a concern |
| N4 bias-field correction | not done | Low priority (scanner × sex independent); revisit if bias-field artefacts suspected |
| Clip to [0.5, 99.5] percentile | skipped for MRI-CORE | Incompatible with pretraining norm; applies to other encoders in the lineup |

### Random-init ViT-B null (2026-04-22)

Encoder: random-init ViT-B (768-d, same architecture and input pipeline as
MRI-CORE — mid-sagittal slice, min-max norm, 3-channel, resize 1024²,
ImageNet mean/std — Gaussian-initialised weights, no pretraining, fixed seed).

| Attribute | Metric | Mean | ±95% CI (NB) | Per-fold scores |
|---|---|---|---|---|
| Sex | AUROC | **0.836** | ±0.039 | 0.831, 0.854, 0.861, 0.827, 0.810 |
| Age (3-bin) | Balanced accuracy | 0.534 | ±0.083 | 0.534, 0.487, 0.529, 0.512, 0.606 |
| Race (3-group) | Balanced accuracy | 0.426 | ±0.064 | 0.396, 0.483, 0.402, 0.428, 0.423 |
| Manufacturer | AUROC | **0.992** | ±0.013 | 0.995, 0.995, 0.980, 0.991, 0.997 |

PCA(2) explained variance: 80.6% (PC1) + 15.6% (PC2) = **96.2% total**.

### Cross-encoder comparison

| Attribute | Metric | MRI-CORE (uncropped) | random_vit_b | Delta |
|---|---|---|---|---|
| Sex | AUROC | 0.931 ±0.031 | 0.836 ±0.039 | +0.095 |
| Age (3-bin) | Balanced accuracy | 0.632 ±0.043 | 0.534 ±0.083 | +0.098 |
| Race (3-group) | Balanced accuracy | 0.513 ±0.071 | 0.426 ±0.064 | +0.087 |
| Manufacturer | AUROC | 0.997 ±0.007 | 0.992 ±0.013 | +0.005 |

Random baseline for 3-class balanced accuracy is 0.333.

## Next steps

- Run non-Duke encoders (Triad-SwinB, MedicalNet) to confirm sex signal
  generalises beyond MRI-CORE's institutional pretraining.
- Add UMAP(2) and per-PC K–S tests (Glocker 2023 style).
- Permutation test on sex AUROC (N=1000 label shuffles).
- Age as continuous regression (R²) rather than 3-bin classification.
