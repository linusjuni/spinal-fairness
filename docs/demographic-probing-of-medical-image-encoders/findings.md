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

1. **Body-extent confound — preliminary evidence against, not yet definitively ruled out.**
   The cropped variant (`mri_core_cropped`) produced sex AUROC 0.936, essentially
   unchanged from 0.931 uncropped, which is reassuring. However, visual inspection of
   the preprocessing preview showed that the bounding-box crop at 5% of max intensity
   is not very aggressive — many scans already had tight FOV, so only a small border
   was removed. The crop also does not zero out background pixels within the bounding
   box; body shape and extent (height, width of anatomy) still vary before the resize
   to 1024². It is therefore possible that the encoder is still reading body-size
   information rather than internal anatomy structure.

   Two stronger ablations are planned (see `idea.md`):
   - **Otsu + `binary_fill_holes` pixel masking** (`mri_core_masked`): zero out all
     non-tissue pixels per-scan using an adaptive threshold. Removes air inside the
     bounding box and handles enclosed cavities (trachea). If sex AUROC stays high,
     the confound is definitively eliminated.
   - **Random-init null encoder**: same ViT-B architecture, no pretraining. If this
     gives AUROC ≈ 0.5 on uncropped images, any preprocessing artefact is ruled out
     as the driver — the signal requires learned weights.

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
| Foreground crop (bbox @ 5% max) | done | Ran 2026-04-22; no effect on probe AUROCs. Pixel-level masking (Otsu + fill-holes) still pending — see `idea.md` |
| In-plane resample to fixed grid | not done | Min-max + resize approximates it; revisit if anatomy scale becomes a concern |
| N4 bias-field correction | not done | Low priority (scanner × sex independent); revisit if bias-field artefacts suspected |
| Clip to [0.5, 99.5] percentile | skipped for MRI-CORE | Incompatible with pretraining norm; applies to other encoders in the lineup |

## Next steps

- **Body-extent confound**: implement `mri_core_masked` (Otsu pixel masking) and
  random-init null encoder — see `idea.md`.
- Run non-Duke encoders (Triad-SwinB, MedicalNet) to confirm sex signal
  generalises beyond MRI-CORE's institutional pretraining.
- Add UMAP(2) and per-PC K–S tests (Glocker 2023 style).
- Permutation test on sex AUROC (N=1000 label shuffles).
- Age as continuous regression (R²) rather than 3-bin classification.
