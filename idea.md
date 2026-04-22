# Ideas / Backlog

## Stronger background removal for FOV confound ablation

**Context**: `mri_core_cropped` uses a bounding-box crop at 5% of max intensity. Sex
AUROC was unchanged (0.931 → 0.936), but the crop is not very aggressive — some air
remains and body shape/size still varies before the resize.

**Idea**: Add an `mri_core_masked` preprocessing variant using Otsu thresholding +
`scipy.ndimage.binary_fill_holes` to generate a per-scan tissue mask, then zero out
all non-tissue pixels before feeding to the encoder. Otsu adapts per-scan and gives
essentially perfect tissue/air separation on MRI without needing SAM or a learned
segmentation model. `binary_fill_holes` handles enclosed air (e.g. trachea).

If sex AUROC stays high with pixel-level masking, the body-extent confound is
definitively ruled out.

## Random-init null encoder

**Context**: Required by sketch.md; the most compelling argument against any
preprocessing confound.

**Idea**: Register a `random_vit_b` encoder — ViT-B with the same architecture as
MRI-CORE but randomly initialised weights, no pretraining. Run the full probe pipeline
on it. If sex AUROC ≈ 0.5, it confirms the signal requires learned features and cannot
be an intensity-histogram or FOV artefact. If AUROC is high, something in the image
statistics alone is driving it.

This is more rigorous than any preprocessing ablation and sidesteps the "was the
masking aggressive enough?" debate entirely.
