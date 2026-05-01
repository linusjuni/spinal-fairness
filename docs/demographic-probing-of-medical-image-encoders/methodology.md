# Methodology

Specifics for each stage of the demographic-probing pipeline. The
high-level flow and pseudocode are in `sketch.md`; this document focuses
on *why* each choice and *what* to tune.

---

## Preprocessing

Duke CSpineSeg is highly anisotropic (~0.5×0.5 mm in-plane, ~4 mm slice).
Forcing isotropy would 8× interpolate the slice axis and invent data.

1. **Orientation** — reorient to canonical RAS (`nibabel.as_closest_canonical`
   or SimpleITK). Non-negotiable.
2. **Resampling**
   - Resample in-plane to a fixed grid (e.g. 1×1 mm on 256×256).
   - Keep native slice thickness. Do NOT force 1 mm isotropic.
   - If the encoder mandates a fixed crop (SAM-Med3D: 128³), resample to
     that crop and accept distortion — diagnostic only.
3. **Intensity normalisation** — MRI has no absolute scale:
   - Clip to [0.5th, 99.5th] percentile.
   - Z-score per volume.
   - Do NOT apply HU windowing. For CT-pretrained encoders, either skip it
     or rescale into the expected range ([-1, 1]) and document the caveat.
4. **Bias-field correction** — N4ITK (SimpleITK `N4BiasFieldCorrection`).
   Cheap; removes a scanner-dependent nuisance.
5. **Foreground crop** — bounding-box around non-zero voxels. **Critical**
   for the FOV confound (see "Confounds" below). Report results with and
   without cropping.

### Current implementation status (MRI-CORE MVP)

The per-encoder preprocessing **deviates** from the recipe above when the
encoder's pretraining mandates a different intensity scheme. MRI-CORE was
pretrained with per-slice min-max to [0, 1] + ImageNet mean/std; feeding
clip + z-score inputs to a ViT tuned for that scheme produces wrong
features. The MRI-CORE pipeline is therefore encoder-specific, not
project-standard. Other encoders (MedicalNet, TotalSpineSeg, Triad) will
use clip + z-score per the spec above.

| Step | MRI-CORE MVP | Notes |
|---|---|---|
| RAS reorient | done | Required for all encoders |
| Mid-sagittal slice selection | done | 2D encoder; 3D encoders take the full RAS volume |
| Per-slice min-max + ImageNet norm | done | Replaces clip + z-score for MRI-CORE only |
| Grayscale → 3-channel, resize to 1024² | done | SAM input requirement |
| Foreground crop | **variant available** (`mri_core_cropped`), not yet run | Body-extent ablation |
| In-plane resample to fixed mm grid | not done | Min-max + resize-to-1024² approximates it; add if cross-scanner resolution variance becomes a concern |
| N4 bias-field correction | not done | Low priority given scanner × sex is independent (χ² p=0.21); revisit if bias-field artefacts are suspected |
| Clip to [0.5, 99.5] percentile | skipped for MRI-CORE | Applies to other encoders in the lineup |

---

## Feature extraction

One table per encoder variant; pick the recipe matching the chosen
encoder. The lineup is three primaries (anatomy match, institution+SSL,
3D scale) plus legacy baselines and controls; see
`encoder-recommendations.md` for the full story.

| Encoder | Role | Layer | Pool | Output |
|---|---|---|---|---|
| **TotalSpineSeg** (nnU-Net) | primary: anatomy match | encoder bottleneck (strip decoder) | `adaptive_avg_pool3d((1,1,1))` | ~320-d (nnU-Net default) |
| **MRI-CORE** ViT-B (SAM-init DINOv2) | primary: institution + SSL / MVP | CLS token | mean across slices (or mid-sagittal only) | 768-d |
| **Triad Swin-B (MAE)** | primary: 3D SSL scale | SwinUNETR-B bottleneck | `adaptive_avg_pool3d((1,1,1))` | 768-d |
| MedicalNet ResNet-50 | legacy baseline | `layer4` (strip `conv_seg`) | `adaptive_avg_pool3d((1,1,1))` | 2048-d |
| RadImageNet R-50 slice-wise | mixed-modality control | global avg pool | mean across slices | 2048-d |
| RadImageNet R-50 mid-slice only | mixed-modality control | global avg pool | — | 2048-d |
| CT-FM | CT null (signal through mismatch?) | encoder output | pooled | varies |
| BiomedCLIP ViT-B/16 slice-wise | robustness probe | CLS token | mean across slices | 512-d |
| DINOv2 ViT-L slice-wise | robustness probe (natural-image) | CLS token | mean across slices | 1024-d |
| Random-init 3D ResNet-18 | mandatory null | `layer4` | `adaptive_avg_pool3d((1,1,1))` | 512-d |
| ImageNet R-50 slice-wise | second null | global avg pool | mean across slices | 2048-d |

Save each set as a numpy array paired with a polars DataFrame of
`series_submitter_id` + demographics. Plan for 3 primaries + MedicalNet
+ the two nulls minimum; add CT-FM and the robustness probes once the
core plot is in.

**For MRI-CORE specifically**: it's a 2D slice-wise encoder. Two
aggregation variants to produce per-volume features: (a) mid-sagittal
slice only, (b) mean-pool across the middle third of sagittal slices.
Compare — Glocker 2023 saw stable probe results across aggregation
choices, which is itself a sanity signal.

**For TotalSpineSeg specifically**: the model is a supervised nnU-Net
(not SSL), so its bottleneck features are narrower than MRI-CORE's or
Triad's. Feature quality caveat: the encoder may not encode demographic
signal simply because the segmentation supervisor never asked it to.
Interpret probe AUROCs on TotalSpineSeg features as a *lower bound*
rather than a clean measurement.

---

## Dimensionality reduction

**PCA(50) → UMAP(2)**, not UMAP on raw features.

Why the PCA step:
- De-noises.
- Removes correlated dims that distort UMAP distances.
- PCA components are interpretable on their own (Glocker 2023 runs K–S
  tests directly on PCs 1–10).

**UMAP hyperparameters for n≈1,254:**

| Parameter | Value | Rationale |
|---|---|---|
| `n_neighbors` | 30 | Default 15 is too local at this n; 30 balances local/global |
| `min_dist` | 0.1 | For plots. Use 0.0 if feeding into HDBSCAN |
| `metric` | `"cosine"` | Deep-learning embeddings; Euclidean defensible after PCA |
| `n_components` | 2 | For plots |
| `random_state` | fixed | Report the seed |

Also produce a 10-dim UMAP for numeric metrics, and a **PaCMAP** or t-SNE
(perplexity=30) plot as a sanity check — UMAP-specific artefacts are a
real risk.

---

## Quantitative metrics

Report per (encoder, attribute) where attribute ∈ {sex, age bin, race,
scanner vendor, field strength}.

### 1. Linear probe AUROC (primary)
- Logistic regression (sklearn, L2, C tuned by inner CV).
- 5-fold stratified CV on the 50-d PCA features.
- Report mean ± 95% CI across folds.
- Balanced accuracy for multi-class (age bin, race).

### 2. Non-linear probe AUROC
- Small MLP: 1 hidden layer, 128 units.
- Detects non-linear encoding that the linear probe would miss.

### 3. Silhouette score
- Attribute labels in 50-d PCA space (or 10-d UMAP).
- Absolute values are low in practice — report delta vs random baseline.

### 4. Clustering agreement
- K-means (k = number of groups), ARI and NMI against true labels.
- Same caveat as silhouette.

### 5. Per-PC K–S test (Glocker 2023 style)
- Two-sample Kolmogorov–Smirnov on each PC 1–10, per protected-group pair.
- Report p-values with Bonferroni / Holm correction.

### Age treatment
Age should also be regressed as **continuous**: linear regression on PCA
features, report R². Cleaner than 3-bin classification and matches
Parikh's MAMA-MIA treatment.

---

## Significance testing

### Permutation test on linear-probe AUROC
1. Permute attribute labels N=1,000 times.
2. Retrain probe each time.
3. Build null distribution of AUROCs.
4. Report empirical p-value.

### Random-feature baseline (mandatory)
Run the full pipeline against a **randomly-initialised 3D ResNet-18** on
the same cohort. If random features reach comparable probe AUROC, the
signal is FOV / intensity-histogram artefact — NOT learned anatomy.

Also compare to **ImageNet ResNet-50 slice-wise** for a second cheap null.

### Institution-fingerprint disentanglement (also mandatory)
MRI-CORE is pretrained on Duke-institutional data. Even though cervical
anatomy was excluded from its pretraining, the encoder may still encode
Duke scanner/protocol fingerprints particularly cleanly — and those
fingerprints may correlate with demographics. To disentangle:

1. Run the probe on **Triad-SwinB** (non-Duke MRI pretraining) and
   **MedicalNet** (non-Duke mixed CT+MRI). These should *not* have a
   Duke acquisition bias.
2. If MRI-CORE leaks demographic signal substantially more than Triad /
   MedicalNet, some fraction of the MRI-CORE signal is acquisition
   fingerprint, not anatomy.
3. Report scanner-vendor and field-strength probes alongside
   demographic probes on all encoders — same confound logic.

---

## Field conventions for interpretation

| Linear-probe AUROC | Interpretation |
|---|---|
| ≥ 0.80 | Strong encoder leakage of the attribute |
| 0.65 – 0.80 | Moderate; investigate FOV / intensity confounds |
| < 0.60, not above random baseline | No evidence of encoding* |

*No evidence of encoding does not imply downstream fairness
(cf. Petersen et al. arXiv:2305.01397).

---

## Confounds to rule out

### FOV — #1 trivial shortcut
- Taller patients and men get larger FOV.
- Scanner vendors have different default FOV.
- **Always report results with and without foreground cropping.** If the
  signal disappears after cropping, the encoder is reading body size, not
  anatomy.

### Scanner / field strength
- CSpineSeg is 63% Siemens / 37% GE.
- Cross-cohort EDA showed race × scanner is NOT confounded (docs/comparison.md).
- But probe scanner AUROC alongside demographics. If scanner clusters
  harder than sex, the story is acquisition-driven.

### Intensity-histogram artefacts
- Vendors produce different intensity distributions even after z-scoring.
- Random-init encoder baseline catches this.

### Institutional acquisition fingerprint (MRI-CORE-specific)
- MRI-CORE was pretrained on Duke clinical data 2016–2020 — the same
  institution, scanners, and era as CSpineSeg.
- This is a feature (cleanest intensity statistics) and a bug (may
  encode Duke acquisition signatures that correlate with demographics).
- Must compare against non-Duke encoders (Triad, MedicalNet) to
  disentangle signal from fingerprint. See significance-testing section.

---

## Why UMAP alone is insufficient

Three failure modes that require the quantitative probe as a companion:

1. UMAP can manufacture clusters from noise at low `n_neighbors`.
2. 2-D UMAP geometry does not preserve high-dim distances faithfully.
3. Groups that visually overlap in UMAP often separate at AUROC > 0.9 in
   the original feature space.

Always report UMAP plot + linear probe AUROC + one auxiliary metric.
