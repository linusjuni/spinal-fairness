# Encoder Recommendations

Candidate encoders for extracting frozen features from Duke CSpineSeg sagittal
T2 cervical spine MRI (n≈1,254 volumes, ~0.5×0.5×4 mm, highly anisotropic).

Revised 2026-04-21 — 2024-25 released the first wave of 3D MRI-native
foundation models (Triad, MRI-CORE, Decipher-MR, 3DINO, MedSAM2). These
reshape the defaults; MedicalNet is no longer the obvious primary.

---

## Recommendation

No single encoder wins on every axis — the right lineup covers three
complementary alignment axes (anatomy, institution+modality+scale,
3D-SSL scale). Run all three as primaries; treat the legacy baselines
as controls.

### Primary — anatomy-matched: TotalSpineSeg encoder

**TotalSpineSeg** (Neuropoly 2024-25).
- nnU-Net trained on SPIDER (lumbar) + Spine Generic multi-subject
  (includes cervical + whole-spine, multi-vendor) + whole-spine datasets
  — ~1,404 spine MRIs total.
- **Only public encoder that has actually seen T2 cervical sagittal MRI
  during training.** Best anatomy fit by a wide margin.
- Caveats: supervised-segmentation objective (features narrow —
  optimized for predicting vertebra/IVD/cord/canal labels, may hide
  features the supervisor didn't ask for); must strip the decoder head
  and pool the bottleneck yourself (no off-the-shelf feature extractor).
  Pretraining scale is ~80× smaller than the SSL alternatives.
- Repo: `neuropoly/totalspineseg`. SPINEPS (sagittal T2 TSE) is the
  second-best spine proxy.

### Primary — MVP / scale + SSL: MRI-CORE

**MRI-CORE** (Mazurowski lab, Duke; arXiv:2506.12186, 2025).
- 2D ViT-B, DINOv2 SSL initialised from SAM weights.
- Pretrained on **Duke-110K**: 6.9 M slices / 116,806 volumes / 107 MRI
  protocols / 2016–2020 at Duke University Medical Center.
- Apache-2.0 weights (`MRI_CORE_vitb.pth`). Same institution / same
  scanners / same era as CSpineSeg, but **cervical anatomy was
  deliberately excluded** from pretraining (Methods: "exclude … brain,
  face, and neck MRIs"; Fig. 1b body-location list has lumbar spine but
  no cervical).
- **Contamination check: clean.** CSpineSeg volumes are not in Duke-110K.
- Distributional alignment story: institution ✅ (Duke Siemens + GE
  matches CSpineSeg's 63/37 mix), modality ✅ (100 % MRI), sequence ✅
  (T2 in the 9-sequence pretraining pool), MSK anatomy ✅
  (lumbar-spine-adjacent), cervical anatomy ❌ (unseen — which is what
  you want for a clean probe).
- **The alignment story vs TotalSpineSeg**: MRI-CORE has institution +
  modality + scale + SSL advantages; TotalSpineSeg has anatomy
  advantage. They are complementary, not interchangeable.
- Ready-to-use feature extraction API — lowest-friction MVP path.
- Repo: `mazurowski-lab/mri_foundation`.

### Primary — 3D SSL at scale: Triad

**Triad Swin-B (MAE)** (Wang et al., arXiv:2502.14064, 2025).
- Native 3D, MRI-only SSL on Triad-131K — largest public 3D MRI
  pretraining corpus. Body-location breakdown not disclosed at repo
  level, so anatomy alignment for cervical spine is uncertain; scale
  and MRI-purity are in its favour.
- SwinUNETR-B + PlainConvUNet backbones, MAE + SimMIM variants.
- Repo: `wangshansong1/Triad`, weights on Google Drive.

### Recommended probe lineup

- **MVP (day 1)**: MRI-CORE on a single mid-sagittal slice per volume →
  PCA → scatter plot. Fewest steps to a first answer.
- **Full probe**: TotalSpineSeg (anatomy-matched) + MRI-CORE
  (institution + SSL) + Triad-SwinB (3D MRI-scale) + MedicalNet
  (reviewer baseline) + CT-FM (CT control) + random-init null.
- No single encoder answers the question alone — the cross-encoder
  comparison is what disentangles "signal from anatomy" from "signal
  from acquisition fingerprint" from "signal from generic intensity
  artefact."

### Legacy baselines (reviewers expect these)

**MedicalNet ResNet-50** (Chen 2019; `resnet_50_23dataset.pth`).
- 3D, fully convolutional, mixed CT + MRI pretraining. Still a common
  bias-probe baseline; cheap to run.
- **Gotcha**: published checkpoints are the *segmentation* model. Strip
  `conv_seg`, apply `adaptive_avg_pool3d((1,1,1))` on `layer4` → 2048-d.
  The paper reports 8 pretraining datasets; the "23 datasets" figure is
  the GitHub-release expansion.

**RadImageNet ResNet-50 slice-wise, mean-pooled** (Mei 2022).
- 2D, 1.35 M CT/MR/US slice pretraining. Weights public — only the
  **dataset** is access-gated; the pretrained weights are linked from the
  README via Google Drive.
- Known normalisation pitfall: replicate grayscale to 3 channels, scale to
  [0, 1] for ResNet50/DenseNet, [-1, 1] for Inception variants. **Do NOT
  apply ImageNet mean/std** — PyTorch ports have gotten this wrong.

### Mandatory nulls for the permutation test

- Randomly-initialised 3D ResNet-18 (or 3D ViT-B to match Triad).
- ImageNet-pretrained ResNet-50 slice-wise + mean pool.
- If random features already cluster by demographics, the signal is
  FOV / intensity-histogram artefact, not learned anatomy.

### Robustness probes (run after MVP works)

- **3DINO-ViT** (npj Dig Med 2025) — 3D-adapted DINOv2 on 70 K MRI + 27 K
  CT + 566 PET; strongest modern multi-modal 3D comparator. Repo:
  `AICONSlab/3DINO`.
- **MedSAM2** (arXiv:2504.03600, 2025) — SAM2-based, 455 K 3D masks (77 K
  MRI, 363 K CT). Supersedes SAM-Med3D on MRI coverage and recency. Repo:
  `bowang-lab/MedSAM2`.
- **Swin-BOB / UK-BOB** (ICCV 2025) — Swin-UNETR on 51 K UK Biobank body
  MRI. UKB-scale modality comparator, body (not spine) anatomy.
- **BrainSegFounder** (arXiv:2406.10395) — MRI-native but brain only.
  Clean "modality match, anatomy mismatch" ablation.
- **BiomedCLIP ViT-B/16 slice-wise** — 2D VLM, trained on PMC figures
  (not clinical scans — expect weaker signal than MRI-CORE / RadImageNet).
- **DINOv2 ViT-L slice-wise** — natural-image only; Baharoon et al.
  (arXiv:2312.02366) show it's a strong frozen baseline under linear
  probing on radiology benchmarks.

### Watch-item

**Decipher-MR** (Yang et al., GE HealthCare; npj Digital Medicine 2026;
arXiv:2509.21249). 3D vision-language model, 200 K MRI series,
**explicitly covers spine**, and the paper already benchmarks on sex/age
prediction from 3D MRI features — literally this project's task. Weights
release status unconfirmed as of 2026-04; check the npj code-availability
statement before planning around it.

### CT-pretrained controls (for "does signal leak through mismatched encoder?")

- **CT-FM** (AIM Harvard 2025) — 148 K IDC CT, pip-installable; modern
  replacement for Models Genesis.
- **VoCo-B / Large-Scale-Medical** (TPAMI 2025) — 160 K volumes,
  Swin-UNETR backbones 31 M – 1.2 B params. Triad is built on VoCo v2,
  so VoCo also gives a clean "same architecture, CT vs MRI pretrain"
  ablation.

---

## Survey — 3D encoders (MRI-native highlighted)

| Model | Year | Pretraining | Modality | Input | Feature | Usable? |
|---|---|---|---|---|---|---|
| **TotalSpineSeg** | 2024-25 | SPIDER + Spine Generic + whole-spine, ~1,404 MRIs | **MRI, incl. cervical sagittal T2** | nnU-Net | bottleneck (strip decoder) | ✅ Primary, anatomy-matched |
| **Triad Swin-B / MAE** | 2025 | Triad-131K | **MRI** (body coverage undisclosed) | 3D crop | SwinUNETR-B | ✅ Primary, 3D scale |
| **Decipher-MR** | 2025 | 200 K MRI series | **MRI incl. spine** | 3D | VL-embed | ⏳ Weights pending |
| **3DINO-ViT** | 2025 | 70 K MRI + 27 K CT + 566 PET | **MRI-heavy** | 3D ViT | ViT-B/L | ✅ Robustness probe |
| **MedSAM2** | 2025 | 455 K masks (77 K MRI, 363 K CT) | Mixed | 3D SAM2 | token | ✅ Supersedes SAM-Med3D |
| **Swin-BOB / UK-BOB** | 2025 | 51 K UKB body MRI | **MRI (body)** | 3D Swin-UNETR | feature map | ✅ UKB-scale comparator |
| **BrainSegFounder** | 2024 | ~41 K UKB brain | **MRI (brain only)** | 3D Swin-UNETR | feature map | ⚠️ Anatomy mismatch |
| **Revisiting-MAE-3D / nnSSL** | 2025 | 44 K brain MRI | **MRI (brain)** | 3D ResEnc | — | ⚠️ Brain only |
| MedicalNet R50 | 2019 | 3DSeg-8 + 23-dataset expand | Mixed CT + MRI | Arbitrary 3D | 2048-d (after pool) | ✅ Legacy baseline |
| SAM-Med3D | 2023-24 | SA-Med3D-140K (22 K vols / 143 K masks) | Mixed CT + MRI + US | 128³ @ 1.5 mm iso | 384-d tokens | ⚠️ Superseded by MedSAM2 |
| CT-FM | 2025 | 148 K IDC CT | CT | 3D | — | ⚠️ CT null |
| VoCo / Large-Scale-Medical | 2024-25 | PreCT-160K | CT-dominated | 3D Swin-UNETR 31 M – 1.2 B | — | ⚠️ CT null |
| Models Genesis | 2019 | **623 LIDC-IDRI CTs** (not LUNA16) | CT (lung only) | 3D | 512-ch bottleneck (needs pool) | ❌ Superseded by CT-FM |
| SuPreM | 2024 | AbdomenAtlas 1.1, 9,262 CT, 25 organs | CT abdomen (HU-windowed) | 3D | varies | ⚠️ CT abdomen |
| STU-Net | 2023 | TotalSegmentator 1,204 CTs (incl. C1–C7) | CT | 3D | varies | ⚠️ CT, but sees vertebrae |
| SegVol | 2024 | 90 K + 6 K CT | CT | 3D ViT | patch tokens | ⚠️ CT |
| Merlin | Nature 2026 | 15,331 abd CT + reports | CT abdomen | 3D | dense | ❌ Wrong anatomy |
| Universal Models (Zhou) | 2024 | 3,410 CTs / 14 datasets / 25 organs | CT abdomen | 3D | — | ❌ Wrong anatomy |

Nothing public is pretrained on *spine MRI specifically* at foundation-model
scale. SPIDER (447 lumbar MRIs) and TotalSpineSeg (1,404 spine MRIs) exist
as segmenters whose encoders can be repurposed. SpineFM is X-ray, not MRI.

---

## Survey — 2D encoders applied slice-wise

Given the ~8× anisotropy, 2D slice-wise is principled, not a fallback.

| Model | Pretraining | Output | Notes |
|---|---|---|---|
| **MRI-CORE** | Duke-110K: 6.9 M slices, 116,806 vols, 18 body locs (lumbar spine yes, cervical/neck excluded), 9 sequences | 768 (ViT-B) | **MVP primary** — Duke institution + 100 % MRI + SSL; cervical anatomy unseen; Apache-2.0 |
| RadImageNet R-50 | 1.35 M CT/MR/US slices | 2048-d | Weights public; grayscale×3 ch, [0, 1] (NOT ImageNet norm) |
| BiomedCLIP ViT-B/16 | 15 M PMC figure-caption pairs | 512 (joint) / 768 (vision) | Trained on paper figures — weaker on clinical DICOM |
| DINOv2 ViT-L | LVD-142M natural images | 1024-d | Strong frozen baseline (Baharoon 2023) |
| Rad-DINO | Chest radiographs | 768-d | Wrong anatomy, right grayscale-radiology texture |
| MedSigLIP / MedGemma 1.5 | CXR + derm + path + ophth (no 3D MRI) | varies | 2D only, no clear advantage over BiomedCLIP for MRI |
| Medical Slice Transformer (Müller-Franzes / Truhn 2025) | Frozen DINOv2 + transformer aggregator | — | Aggregation recipe, not an encoder — reuse DINOv2 features directly |
| MedCLIP / PubMedCLIP | Small radiology corpora | 512 | Less benchmarked than BiomedCLIP; in BiomedCLIP's own paper sometimes underperforms vanilla CLIP |

**Aggregation 2D → 3D embedding** (in order of sophistication):
1. Mean pool across slices.
2. Mid-sagittal slice only.
3. Mean + max concat.
4. Learned attention pool (requires minimal training).
5. Frozen transformer aggregator (Medical Slice Transformer recipe).

For the diagnostic probe: **mean pool** + **mid-sagittal-only** as a
sanity check. Two cheap variants, compare.

---

## Gotchas

- **MedicalNet checkpoints are segmentation-head models.** Strip `conv_seg`,
  add your own `adaptive_avg_pool3d((1,1,1))` + flatten on `layer4` → 2048-d.
  Use per-volume z-score (dataset-specific normalisation; no global mean/std
  exists). Freeze BN (`model.eval()` is essential). Pad input spatial dims
  to multiples of 16 to avoid skip-connection shape mismatches. The paper
  reports 8 pretraining datasets; the "23 datasets" number is only in the
  GitHub README's later release.

- **SAM-Med3D / MedSAM2 expect ~1.5 mm isotropic, 128³ crops.** Sagittal T2
  cervical (12–18 slices at 4 mm) will through-plane-interpolate
  substantially before reaching the encoder. Acceptable for the diagnostic
  but note the caveat. Same problem affects SuPreM, VoCo, Triad.

- **SAM-Med3D pretraining size**: the 21 K volumes / 131 K masks / 247
  anatomies figures are the *2023 paper*. The current checkpoint
  (`SA-Med3D-140K`) reports 22 K / 143 K. Cite whichever matches the
  checkpoint you download.

- **SAM-Med3D token dim**: `build_sam3D_vit_b` uses embed_dim = 384
  (not the ViT default 768); 128³ input with patch 16³ yields 8³ = 512
  tokens × 384-d, flattening to ~196 K-d. Use mean-pool over tokens →
  384-d or the CLS-equivalent token.

- **RadImageNet normalisation** is [0, 1] (Keras/TF ResNet/DenseNet) or
  [-1, 1] (Inception variants), grayscale replicated to 3 channels —
  **NOT ImageNet mean/std**. Public PyTorch ports have gotten this wrong;
  confirm against the original TF preprocessing.

- **BiomedCLIP was trained on PMC figures**, many of them multi-panel,
  colour-mapped, or annotated. Expect lower signal than MRI-CORE or
  RadImageNet on raw clinical grayscale slices (medRxiv reports this).

- **CT-pretrained 3D models** (SuPreM / Merlin / SegVol / Models Genesis
  / CT-FM) implicitly assume HU-windowed inputs. Feeding z-scored MRI
  through them is only defensible as a "does signal leak through a
  mismatched encoder?" probe.

- **FOV confound** — taller patients and men get larger FOV; vendors have
  different default FOV. Always report results both with and without
  foreground cropping (see `methodology.md`).

- **Scanner-demographic correlation** — Duke CSpineSeg is 63 % Siemens /
  37 % GE. Probe scanner and field-strength alongside demographics; if
  scanner clusters harder than sex, the story is acquisition-driven, not
  anatomy.

- **MRI-CORE Duke-internal pretraining is a double-edged sword.** Same
  institution and scanners means features should encode this data
  precisely. But it also means MRI-CORE may capture Duke acquisition
  fingerprints particularly cleanly, potentially inflating apparent
  demographic signal if vendor/protocol choices correlate with patient
  demographics. This is exactly why the non-Duke encoder comparators
  (Triad, MedicalNet) and the random-init null are non-negotiable —
  disentangling "encoder knows demographics" from "encoder knows Duke
  scanner fingerprints that correlate with demographics" requires both.
