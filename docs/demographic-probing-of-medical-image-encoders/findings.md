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

| Attribute | Metric | MRI-CORE (uncropped) | random_vit_b | Delta (MRI-CORE − random) |
|---|---|---|---|---|
| Sex | AUROC | 0.931 ±0.031 | 0.836 ±0.039 | +0.095 |
| Age (3-bin) | Balanced accuracy | 0.632 ±0.043 | 0.534 ±0.083 | +0.098 |
| Race (3-group) | Balanced accuracy | 0.513 ±0.071 | 0.426 ±0.064 | +0.087 |
| Manufacturer | AUROC | 0.997 ±0.007 | 0.992 ±0.013 | +0.005 |

Random baseline for 3-class balanced accuracy is 0.333.

## Interpretation

**Sex (0.931 → 0.836 under random-init).** MRI-CORE is above the 0.80
strong-leakage threshold, but the random-init ViT-B null retains 0.836 —
untrained features alone recover 90 % of the signal from raw pixel
statistics. The pretraining contribution is a real but modest +0.095
AUROC; the dominant driver of MRI-CORE's sex signal is image-level
(FOV, body silhouette, intensity histogram, noise texture), not learned
cervical anatomy. PCA(2) scatter on MRI-CORE features showed no visible
separation, confirming the signal lives beyond the top 2 components
there; on random features PCA(2) explains 96 % of variance — those
features have collapsed to a handful of low-order image statistics.

**Age (0.632 → 0.534 under random-init).** Moderate signal above chance
(0.333 for 3-class balanced accuracy). Random features already get 0.534,
so a large share of the MRI-CORE age signal sits in raw pixel statistics;
the pretraining contribution is ~+0.10 BA.

**Race (0.513 → 0.426 under random-init).** Weak. MRI-CORE upper bound
(mean + CI) is below the 0.65 "investigate confounds" threshold. Random
features land above chance (0.426) but well below MRI-CORE — whatever
little race signal exists is mostly in raw pixel statistics.

**Manufacturer (0.997 → 0.992).** Essentially unchanged under random-init.
Scanner fingerprint lives entirely in raw pixel statistics (distinct vendor
intensity histograms and noise patterns), as expected — and this serves
as a healthy sanity check that the probe pipeline runs end-to-end on
random features.

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

1. **Body-extent confound — confirmed as the dominant signal source.**
   "Body extent" is three nested sub-confounds: body-to-image *ratio* (how much
   of the frame is tissue), silhouette *shape* (the body contour itself), and
   background *intensity/noise* statistics. Two ablations have now been run.

   **`mri_core_cropped` — inconclusive.** A bbox crop at 5 % of max intensity
   + resize to 1024² was meant to test the fill-ratio sub-confound only. Sex
   AUROC moved +0.005 (0.931 → 0.936), but the intervention turned out to be
   near no-op in practice: visual inspection of
   `outputs/probe/preprocessing_preview.png` showed cropped network inputs
   with large black regions around the body silhouette, nearly identical to
   the uncropped versions. Bbox cropping only removes rows/columns entirely
   below threshold — a loose constraint because anatomy is non-rectangular
   (tapering head / neck outline) so a tight bbox still encloses lots of air,
   and any single above-threshold pixel in a row (nose tip, noise spike)
   preserves the full width of background around it. The fill-ratio
   normalisation this variant was supposed to deliver barely happened. Square
   resize also introduces a new sex-linked stretch artefact (men tend to have
   wider necks per unit height), so the crop may have traded one confound
   for another. A near no-op can't refute the confound it's nominally testing;
   aspect-preserving letterbox resize would be the cleaner input-level
   ablation if we revisit.

   **`random_vit_b` — confirms the body-extent family dominates.** A ViT-B
   with the same architecture and input pipeline as MRI-CORE but no
   pretraining still gets sex AUROC 0.836 (±0.039) — 90 % of MRI-CORE's
   0.931. Random weights cannot see anatomy by construction, so whatever
   the random ViT is reading must be raw pixel statistics: FOV, body
   silhouette, intensity histogram, noise texture — i.e. the body-extent
   confound family. The pretraining contribution on top of this baseline
   is ~+0.10 AUROC, real but clearly secondary. Rather than ruling the
   confound out, the null provides strong evidence that it drives most of
   the observed signal.

   Otsu + `binary_fill_holes` pixel masking was considered and dropped: it
   introduces a hard silhouette edge that is out-of-distribution for
   MRI-CORE (pretrained on raw MRI with air) and does not touch the
   silhouette-shape signal that the random-init result now identifies as
   substantial.

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

## Next steps

- **Aspect-preserving letterbox resize.** Redo the crop ablation with a
  letterbox-pad to 1024² instead of a square stretch, so rectangular bboxes
  don't pick up an anisotropic-stretch confound. Run the variant on both
  `mri_core` and `random_vit_b`. If the MRI-CORE minus random gap widens
  under letterbox, we're recovering anatomy signal that shared body-extent
  noise was previously masking.
- **Inspect what the random ViT is actually reading.** PCA(2) explains
  96 % of variance on random features; PC1 and PC2 loadings should trace
  back to interpretable image statistics. Correlate PC1 / PC2 with
  body-extent proxies (nonzero-pixel count, bbox area, intensity mean,
  centroid) on the raw slices — if one proxy explains most of PC1, we've
  identified the dominant confound explicitly.
- Run non-Duke encoders (Triad-SwinB, MedicalNet). The random-init null
  says pretraining contributes ~+0.10 AUROC of sex signal on top of pixel
  statistics; non-Duke encoders should say whether that +0.10 is a general
  MRI feature or Duke-specific acquisition fingerprint.
- Add UMAP(2) and per-PC K–S tests (Glocker 2023 style).
- Permutation test on sex AUROC (N=1000 label shuffles).
- Age as continuous regression (R²) rather than 3-bin classification.
