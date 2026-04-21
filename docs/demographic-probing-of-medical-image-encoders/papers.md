# Key Literature

The diagnostic pipeline (encode → PCA/UMAP → check demographic clustering)
is an instance of a well-established probe. Read in this order.

---

## The canonical precedent — what this project extends

### Gichoya et al. 2022 — "Reading Race"
**AI recognition of patient race in medical imaging: a modelling study.**
*Lancet Digital Health* 4(6):e406-e414, 2022.

CNNs predict self-reported race from chest X-rays, mammograms, CT, and
**cervical spine radiographs** at high AUC even when images are degraded
or cropped to the point where radiologists cannot. This project's MRI probe
is a direct modality extension.

### Glocker et al. 2023 — template to copy almost verbatim
**Risk of Bias in Chest Radiography Deep Learning Foundation Models.**
*Radiology: AI* 5(6):e230060, 2023.

Exact methodological template for this project:
- Frozen foundation-model embeddings → PCA projection.
- Per-subgroup distribution analysis.
- **Two-sample Kolmogorov–Smirnov tests** on each principal component.

Code: `biomedia-mira/cxr-foundation-bias`. Read before writing any code.

### Glocker et al. 2023 — the full three-part probe
**Algorithmic encoding of protected characteristics in chest X-ray disease
detection models.**
*eBioMedicine* 89:104467, 2023.

Proposes the probe triad adopted here:
1. Test-set resampling.
2. Multitask-learning probe.
3. Unsupervised exploration of feature representations.

Code: `biomedia-mira/chexploration`.

---

## Benchmarks and companion reads

### FairMedFM — the systematic benchmark
**FairMedFM: Fairness Benchmarking for Medical Imaging Foundation Models.**
Jin et al., *NeurIPS D&B* 2024. arXiv:2407.00983.

Benchmarks 20 foundation models across 17 datasets with linear-probe
sensitive-attribute prediction. Gives concrete numbers to compare against
when reporting probe AUROCs.

### Jones et al. — temper the interpretation
**Are demographically invariant models and representations in medical
imaging fair?** arXiv:2305.01397, 2023.

Demographic invariance is *neither necessary nor sufficient* for fairness.
Important caveat: finding demographic signal in embeddings does not imply
the segmentation model is unfair, and removing it does not imply fairness.

### Ioannou et al. — closest MRI analogue
**Investigating Demographic Bias in Brain MRI Segmentation.**
arXiv:2510.17999, 2025.

Brain MRI, not spine, but same methodological family.

---

## Project-internal prior work (supervisor)

### Parikh et al. — MAMA-MIA "Biased Ruler"
**Investigating Label Bias and Representational Sources of Age-Related
Disparities in Medical Segmentation.** arXiv:2511.00477, 2025.

**Who Does Your Algorithm Fail? Investigating Age and Ethnic Bias in the
MAMA-MIA Dataset.** arXiv:2510.27421, 2025.

The supervisor's prior work that this project's methodology should be
compatible with. Age treatment (continuous regression, not just 3-bin
classification) follows this.

---

## Interpretive logic — why UMAP alone is weak evidence

Three reasons to never report UMAP clustering without a quantitative probe:

1. **UMAP can manufacture structure.** At low `n_neighbors` the manifold
   approximation creates apparent clusters from noise.
2. **Geometry is not faithful.** Pairwise distances in the 2-D embedding
   do not match high-dimensional distances (see "Understanding UMAP",
   pair-code.github.io).
3. **Linear separability ≠ visible separation.** Groups that overlap in
   the 2-D projection routinely separate at AUROC > 0.9 in the original
   feature space.

**Canonical quantitative probe:** logistic regression on frozen embeddings,
predicting the protected attribute, reported as AUROC / balanced accuracy
on held-out CV folds. Used by every paper above.

Stronger alternatives: non-linear probes (small MLP), mutual-information
estimators (MINE, InfoNCE bounds), per-PC K–S tests.

Standard practice: report all three — UMAP plot (qualitative), linear
probe AUROC (primary quantitative), one auxiliary (ARI/NMI or per-PC K–S).

---

## Field conventions for "significant evidence of encoding"

From Gichoya 2022 and FairMedFM thresholds:

| Linear-probe AUROC | Interpretation |
|---|---|
| ≥ 0.80 | Strong evidence encoder leaks the attribute |
| 0.65 – 0.80 | Moderate; check FOV / intensity confounds |
| < 0.60, not above random-feature baseline | No evidence of encoding* |

*No evidence of encoding ≠ fairness downstream — see Jones et al.

---

## Links

- [Gichoya et al., Lancet Digital Health 2022](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(22)00063-2/fulltext)
- [Glocker et al., Radiology: AI 2023](https://pubs.rsna.org/doi/full/10.1148/ryai.230060)
- [Glocker et al., eBioMedicine 2023](https://www.thelancet.com/journals/ebiom/article/PIIS2352-3964(23)00032-4/fulltext)
- [biomedia-mira/cxr-foundation-bias](https://github.com/biomedia-mira/cxr-foundation-bias)
- [biomedia-mira/chexploration](https://github.com/biomedia-mira/chexploration)
- [FairMedFM arXiv:2407.00983](https://arxiv.org/html/2407.00983v3)
- [Jones et al. arXiv:2305.01397](https://arxiv.org/html/2305.01397v3)
- [Parikh et al. arXiv:2511.00477](https://arxiv.org/abs/2511.00477)
- [Parikh et al. arXiv:2510.27421](https://arxiv.org/abs/2510.27421)
- [Ioannou et al. arXiv:2510.17999](https://arxiv.org/html/2510.17999v2)
- [Understanding UMAP (pair-code)](https://pair-code.github.io/understanding-umap/)
