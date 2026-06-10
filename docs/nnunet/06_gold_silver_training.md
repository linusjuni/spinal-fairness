# 06 — Gold / Silver Label Experiment

> **Status (2026-06-10):**
> - **Gold (Dataset002): fully evaluated + biased ruler done.** All 10 folds trained. Ensemble
>   selected (CV macro Dice 0.8831). Predictions on 76 gold test images: VB 0.914, Disc 0.862,
>   Macro 0.888. Biased ruler experiment complete and re-aggregated under the binarized DIR/DPD
>   definition (Run 8, `20260607_210826`): silver ruler **saturates** (DIR ≡ 1.0 at 0.8 — the
>   gap hides above the "good enough" bar), so the real signal is in the continuous tests —
>   **silver 11/63 FDR-significant, all age; gold 0/63.** The old mean-based "70–86% narrowing"
>   claim is superseded — see `fairness-runs.md` Run 8 and `dpd-dir-redefinition.md`.
>   *Provenance note:* `2d` fold 1 crashed at epoch 959 (transient `OSError: Stale file handle`)
>   and was finished via a resume/validation step — it had converged (poly-LR ≈ 0, EMA Dice
>   plateaued ~0.89), so no effect on results.
> - **Silver (Dataset003): fully evaluated ✅.** All 10 folds trained. Ensemble selected (CV
>   macro Dice 0.9745). Predictions on 76 gold test images: VB 0.921, Disc 0.872, Macro 0.897.
>   Bias amplification analysis complete (Run 9, `20260609_163752`): **no bias amplification** —
>   silver-trained DIRs ≈ mixed-trained across all groupings; gold-trained is sometimes *worse*
>   on disc fairness (race_wbo disc DIR 0.813 vs silver 0.875). 0 FDR-significant tests on all
>   three rulers.
>   *Provenance note:* 3d_fullres folds 1 and 4 originally crashed (glibc/DA-worker crash) and
>   were retrained with `nnUNet_n_proc_DA=1`.

## Pipeline at a Glance

How the three datasets are built, how each label set is produced, and which models feed
which experiment. All model-vs-model comparisons are scored on the **same 76-case gold test
set** (`split_v3_gold`, expert labels) so that differences are attributable to training
labels — not to the test data.

```mermaid
flowchart TD
    CS["CSpineSeg<br/>1,255 release → 1,254 after exclusions<br/>→ 1,142 analysis cohort (50/50 M/F downsample)<br/>each image is labelled gold OR silver, never both"]:::src

    CS --> GOLD["Gold labels — expert annotation<br/>448-exam pool"]:::gold
    CS --> SILV["Silver labels — Zhou et al. nnU-Net<br/>(trained on gold, run on unlabelled cases)<br/>694-exam pool"]:::silver

    subgraph TR["Sex-balanced training sets — split_v3 family (50/50 M/F)"]
      direction LR
      DS1["Dataset001 — Mixed<br/>798 train = 318 gold + 480 silver<br/>(production-realistic)"]:::mixed
      DS2["Dataset002 — Gold<br/>288 train, expert only"]:::gold
      DS3["Dataset003 — Silver<br/>450 train, auto only"]:::silver
    end

    GOLD --> DS1
    SILV --> DS1
    GOLD --> DS2
    SILV --> DS3

    DS1 --> M1["Model M1 (mixed)<br/>nnU-Net ResEncUNetL ensemble"]:::mixed
    DS2 --> M2["Model M2 (gold)"]:::gold
    DS3 --> M3["Model M3 (silver)"]:::silver

    M2 -- "predict on the 76 gold-test images" --> GENS["Generated silver labels<br/>= M2 predictions<br/>(the 'silver ruler')"]:::silver

    subgraph EXP["Experiments — model-vs-model comparisons all use the 76-case gold test set (expert labels)"]
      direction TB
      E1["E1 · Global fairness audit<br/>M1 vs all labels, full test set (228)<br/>Is a realistically-trained model fair?"]:::exp
      E2["E2 · Biased ruler<br/>M1 on the 76 images, scored twice<br/>Does the choice of ruler hide/inflate the gap?"]:::exp
      E3["E3 · Bias amplification<br/>M2 vs M3, scored vs gold labels<br/>Does silver-label training widen gaps?"]:::exp
      E4["E4 · Mixed vs gold training<br/>M1 vs M2, scored vs gold labels<br/>Does mixing silver into training hurt fairness?"]:::exp
    end

    M1 --> E1
    M1 --> E2
    GOLD -. "Ruler A (true)" .-> E2
    GENS -. "Ruler B (generated silver)" .-> E2
    M1 --> E4
    M2 --> E3
    M2 --> E4
    M3 --> E3

    classDef src fill:#f3f3f3,stroke:#999,color:#000
    classDef gold fill:#f6e7b0,stroke:#b8941f,color:#000
    classDef silver fill:#dfe3e6,stroke:#8a9199,color:#000
    classDef mixed fill:#cfe2f3,stroke:#3d6fa5,color:#000
    classDef exp fill:#d9ead3,stroke:#4e8f43,color:#000
```

### How each label set is produced

| Label set | Provenance | Used as |
|---|---|---|
| **Gold** | Expert manual annotation (original CSpineSeg) | Training labels for Dataset002; ground-truth reference (Ruler A) for E2–E4 |
| **Silver (original)** | Zhou et al. trained an nnU-Net on the gold development set and ran it on the unannotated cases | Training labels for Dataset003 and the silver portion of Dataset001 |
| **Generated silver** | **M2 (our gold-trained model) predicting on the 76 gold-test images** | The "silver ruler" (Ruler B) for the biased-ruler experiment E2 only |

The *generated silver* labels mirror how the *original* silver labels were created — a
gold-trained nnU-Net applied to images it never saw in training — but produced on our own
split so the 76 test cases are guaranteed unseen.

### How this differs from the MAMA-MIA "biased ruler"

Our biased-ruler setup adapts Aditya et al.'s MAMA-MIA design to a dataset where gold and
silver labels never co-exist on the same image.

```mermaid
flowchart TB
    subgraph MM["Aditya et al. — MAMA-MIA (breast DCE-MRI)"]
      direction TB
      MMI["Every image carries BOTH labels"]
      MMI --> MMG["Gold (expert)"]:::gold
      MMI --> MMS["Silver (independent nnU-Net,<br/>external training data)"]:::silver
      MMG -. "ruler A" .-> MMC{{"Direct ruler comparison<br/>same image, two pre-existing references"}}
      MMS -. "ruler B" .-> MMC
    end

    subgraph OU["This work — CSpineSeg (cervical MRI)"]
      direction TB
      OI["Every image carries EITHER gold OR silver"]
      OI --> OG["Gold (expert)"]:::gold
      OG --> OM2["Train Dataset002 (gold-only)"]:::gold
      OM2 -- "predict on the 76 gold-test images" --> OGS["Generated silver labels"]:::silver
      OG -. "ruler A" .-> OC{{"Ruler comparison on the SAME 76 images<br/>gold labels vs generated silver"}}
      OGS -. "ruler B" .-> OC
    end

    classDef gold fill:#f6e7b0,stroke:#b8941f,color:#000
    classDef silver fill:#dfe3e6,stroke:#8a9199,color:#000
```

- **MAMA-MIA:** each image already has both an expert (gold) and an automated (silver)
  reference, so the two rulers can be compared directly on every image.
- **CSpineSeg:** each image has only one label tier. We therefore *generate* the second
  ruler by predicting with the gold-trained M2 on the 76 gold-test images. Both rulers then
  exist for the same images, and any difference in the observed fairness gap is the pure
  ruler effect.
- **Why not reuse Zhou et al.'s silver model?** Its weights are unavailable, and its training
  split may overlap our gold test cases — a leakage risk. M2 is trained on our split, which
  excludes the test cases by design.

## Three Models, Three Roles

| Dataset | Train cases | Labels | Split file | Role | Status |
|---|---|---|---|---|---|
| `Dataset001_CSpineSeg` | 798 (318 gold + 480 silver) | Mixed | `split_v3` | **Global fairness audit** | Trained |
| `Dataset002_CSpineSeg_Gold` | 288 | Expert (gold) | `split_v3_gold` | **Bias amplification baseline** + **biased ruler** (predictions on gold test images serve as generated silver labels) | **Evaluated ✅** |
| `Dataset003_CSpineSeg_Silver` | 450 | Auto-generated (silver) | `split_v3_silver` | **Bias amplification** — compare against gold-trained | **Evaluated ✅** |

All three use sex-balanced cohorts (50/50 M/F in every split).

### Dataset001 — Global fairness audit

Dataset001 is the "production-realistic" model: trained on all available labels without
distinguishing quality. This is what someone would actually deploy. It is evaluated on
the full test set (228 cases) against all labels, and compared to the published baseline
(Zhou et al.). The main fairness analysis — demographic performance gaps across race,
age, sex — uses this model. See `05_model_selection.md`.

### Dataset002 — Biased ruler experiment

In MAMA-MIA, every image had both a gold and a silver label, so the ruler comparison was
direct: the silver labels came from an independent nnU-Net trained on external data.
CSpineSeg images have either gold or silver — not both.

The adapted approach: use Dataset002 (gold-trained) to generate predictions on the gold
test images. Those predictions serve as the "generated silver labels." This mirrors how the
original silver labels were created (Zhou et al. trained an nnU-Net on the gold development
set and applied it to unannotated cases). We cannot use their original model because (a)
the weights are not available, and (b) their training set uses a different split, so some of
our gold test cases may have been in their training data — a leakage risk. Dataset002 avoids
this since it is trained on our own split, which excludes the test cases by design.

We then evaluate Dataset001 (the mixed model) against both rulers on the same 76 gold test
images:

- Evaluate Dataset001 against **gold labels** → true performance
- Evaluate Dataset001 against **Dataset002's predictions** (generated silver) → observed performance

Any difference in the fairness gap between the two evaluations is the pure ruler effect:
same model, same images, different reference labels. Crucially, Dataset001 and Dataset002
are independently trained models, so Dice ≠ 1.

> **Note:** Evaluating Dataset001 against its own predictions would give Dice = 1.0 for every
> case — that approach is degenerate and was rejected.

### Dataset002 — Bias amplification baseline

Dataset002 is trained on gold (expert) labels only. It has two roles:

1. **Bias amplification baseline** — compare its fairness gaps against Dataset003
   (silver-trained) on the same gold test set. If silver-trained has wider gaps, silver
   labels amplify bias through training.
2. **Silver label generator** for the biased ruler experiment (see above).

### Dataset003 — Bias amplification

Dataset003 is trained on silver (auto-generated) labels only. It is compared against
Dataset002 on the **same gold test set** (76 cases) to isolate whether silver training
labels widen demographic performance gaps (cf. Parikh et al. MAMA-MIA Experiment 4:
66% fairness gap widening, DIR dropping below 0.80).

## Evaluation Design

| Experiment | Model | Eval reference | What it shows |
|---|---|---|---|
| Global fairness audit | Dataset001 (mixed) | All test labels (228) | How fair is a realistically-trained model? |
| Biased ruler | Dataset001 (mixed) | Gold labels vs Dataset002's predictions (generated silver) on gold test images (76) | Does the choice of ruler inflate/deflate the observed fairness gap? |
| Bias amplification | Dataset002 (gold) vs Dataset003 (silver) | Gold test labels (76) | Does training on silver labels widen demographic gaps? |
| Mixed vs gold training | Dataset001 (mixed) vs Dataset002 (gold) | Gold test labels (76) | Does including silver labels in training hurt fairness? |

The gold test set (76 cases from `split_v3_gold`) is the apples-to-apples reference for
all pairwise model comparisons — same test cases, same expert ground truth, different
training labels. For the biased ruler experiment, Dataset002's predictions on these same
76 images serve as the generated silver labels (since CSpineSeg images have either gold
or silver labels, not both — unlike MAMA-MIA).

---

## Step 1 — Prepare Dataset (done 2026-05-07)

`src/nnunet/__init__.py` has a `DATASETS` registry mapping dataset ID to name and split
version. Both `prepare_dataset.py` and `write_splits.py` accept `--dataset-id`:

```bash
# Gold dataset
uv run -m src.nnunet.prepare_dataset --dataset-id 2

# Silver dataset
uv run -m src.nnunet.prepare_dataset --dataset-id 3
```

This creates:
```
$nnUNet_raw/
├── Dataset002_CSpineSeg_Gold/
│   ├── imagesTr/   (288 train + 44 val cases, symlinked)
│   ├── labelsTr/   (expert labels only)
│   ├── imagesTs/   (76 test cases)
│   └── dataset.json
└── Dataset003_CSpineSeg_Silver/
    ├── imagesTr/   (450 train + 66 val cases, symlinked)
    ├── labelsTr/   (auto-generated labels only)
    ├── imagesTs/   (138 test cases)
    └── dataset.json
```

---

## Step 2 — Plan and Preprocess (done)

CPU only (~10–20 min each). Run on login node:

```bash
uv run --env-file .env nnUNetv2_plan_and_preprocess -d 2 --verify_dataset_integrity
uv run --env-file .env nnUNetv2_plan_and_preprocess -d 3 --verify_dataset_integrity
```

---

## Step 3 — Write CV Splits (done)

Same stratified 5-fold logic as `write_splits.py`, split version is derived from `--dataset-id`:

```bash
uv run -m src.nnunet.write_splits --dataset-id 2
uv run -m src.nnunet.write_splits --dataset-id 3
```

Writes `splits_final.json` to each preprocessed dataset directory.

---

## Step 4 — Submit Training Jobs (done)

Same `jobs/train.sh` template, just different dataset IDs. Train both 2d and 3d_fullres
for consistency with Dataset001 — decide on best config per dataset after training via
`find_best_configuration`:

```bash
# Dataset002 — Gold (10 jobs, smaller dataset so faster)
for FOLD in 0 1 2 3 4; do
  sed "s/TPLCONFIG/2d/g;        s/TPLFOLD/${FOLD}/g" jobs/train.sh \
    | sed 's/DATASET_ID=1/DATASET_ID=2/' | bsub
  sed "s/TPLCONFIG/3d_fullres/g; s/TPLFOLD/${FOLD}/g" jobs/train.sh \
    | sed 's/DATASET_ID=1/DATASET_ID=2/' | bsub
done

# Dataset003 — Silver
for FOLD in 0 1 2 3 4; do
  sed "s/TPLCONFIG/2d/g;        s/TPLFOLD/${FOLD}/g" jobs/train.sh \
    | sed 's/DATASET_ID=1/DATASET_ID=3/' | bsub
  sed "s/TPLCONFIG/3d_fullres/g; s/TPLFOLD/${FOLD}/g" jobs/train.sh \
    | sed 's/DATASET_ID=1/DATASET_ID=3/' | bsub
done
```

Or add a `--dataset-id` parameter to `train.sh` and create a `submit_gold_silver.sh`.

---

## Step 5 — Predict, Ensemble, Postprocess (done 2026-06-05 for DS002; 2026-06-10 for DS003)

### Dataset002

Predictions run on interactive GPU node (`a100sh`), both configs in parallel. Ensemble and
postprocessing run on login node.

```bash
# GPU: interactive node (two sessions in parallel)
uv run nnUNetv2_predict -d Dataset002_CSpineSeg_Gold \
    -i "${nnUNet_raw}/Dataset002_CSpineSeg_Gold/imagesTs" \
    -o "${nnUNet_results}/Dataset002_CSpineSeg_Gold/predictions_test_2d" \
    -f 0 1 2 3 4 -tr nnUNetTrainerWandB -c 2d -p nnUNetResEncUNetLPlans --save_probabilities

uv run nnUNetv2_predict -d Dataset002_CSpineSeg_Gold \
    -i "${nnUNet_raw}/Dataset002_CSpineSeg_Gold/imagesTs" \
    -o "${nnUNet_results}/Dataset002_CSpineSeg_Gold/predictions_test_3d_fullres" \
    -f 0 1 2 3 4 -tr nnUNetTrainerWandB -c 3d_fullres -p nnUNetResEncUNetLPlans --save_probabilities

# CPU (after predict jobs finish): find_best_config → ensemble → postprocess
bash jobs/ensemble_and_postprocess.sh 2
```

`find_best_configuration` result: ensemble (2d + 3d_fullres) wins (CV macro Dice 0.8831 vs
0.8819 for 2d alone). No postprocessing applied (keep-largest destroys multi-vertebra
structures, same as Dataset001).

Output: `$nnUNet_results/Dataset002_CSpineSeg_Gold/predictions_test_pp/` (76 files)

### Dataset003

Dataset003 predicts on Dataset002's `imagesTs` (the same 76 gold test images), not its own
138-case silver test set. This keeps the bias amplification comparison on identical cases.
Output dirs use `predictions_gold_test_*` to distinguish from any future silver-test-set run.

```bash
# GPU: interactive node (two sessions in parallel)
uv run nnUNetv2_predict -d Dataset003_CSpineSeg_Silver \
    -i "${nnUNet_raw}/Dataset002_CSpineSeg_Gold/imagesTs" \
    -o "${nnUNet_results}/Dataset003_CSpineSeg_Silver/predictions_gold_test_2d" \
    -f 0 1 2 3 4 -tr nnUNetTrainerWandB -c 2d -p nnUNetResEncUNetLPlans --save_probabilities

uv run nnUNetv2_predict -d Dataset003_CSpineSeg_Silver \
    -i "${nnUNet_raw}/Dataset002_CSpineSeg_Gold/imagesTs" \
    -o "${nnUNet_results}/Dataset003_CSpineSeg_Silver/predictions_gold_test_3d_fullres" \
    -f 0 1 2 3 4 -tr nnUNetTrainerWandB -c 3d_fullres -p nnUNetResEncUNetLPlans --save_probabilities

# CPU: ensemble → postprocess (find_best_config already run separately)
ENS="${nnUNet_results}/Dataset003_CSpineSeg_Silver/ensembles/ensemble___nnUNetTrainerWandB__nnUNetResEncUNetLPlans__2d___nnUNetTrainerWandB__nnUNetResEncUNetLPlans__3d_fullres___0_1_2_3_4"
uv run nnUNetv2_ensemble \
    -i "${nnUNet_results}/Dataset003_CSpineSeg_Silver/predictions_gold_test_2d" \
       "${nnUNet_results}/Dataset003_CSpineSeg_Silver/predictions_gold_test_3d_fullres" \
    -o "${nnUNet_results}/Dataset003_CSpineSeg_Silver/predictions_gold_test_ensemble" -np 8
uv run nnUNetv2_apply_postprocessing \
    -i  "${nnUNet_results}/Dataset003_CSpineSeg_Silver/predictions_gold_test_ensemble" \
    -o  "${nnUNet_results}/Dataset003_CSpineSeg_Silver/predictions_gold_test_pp" \
    -pp_pkl_file "${ENS}/postprocessing.pkl" -plans_json "${ENS}/plans.json" -np 8
```

`find_best_configuration` result: ensemble (2d + 3d_fullres) wins (CV macro Dice 0.9745 vs
0.9728 for 2d alone). No postprocessing applied (same reason as DS001/DS002). Note: CV Dice
is high because the model is trained and validated on silver labels; test Dice against gold
labels is ~0.897 (see Step 6).

Output: `$nnUNet_results/Dataset003_CSpineSeg_Silver/predictions_gold_test_pp/` (76 files)

## Step 6 — Evaluate

### Dataset002 on gold test set (done 2026-06-05)

`labelsTs/` symlinks created for all 76 gold test cases pointing to expert segmentations.

```bash
source .env
uv run --env-file .env -m src.fairness.evaluate \
    --predictions "${nnUNet_results}/Dataset002_CSpineSeg_Gold/predictions_test_pp" \
    --references  "${nnUNet_raw}/Dataset002_CSpineSeg_Gold/labelsTs" \
    --mapping     "${nnUNet_raw}/Dataset002_CSpineSeg_Gold/case_id_mapping.json" \
    --output      "${nnUNet_results}/Dataset002_CSpineSeg_Gold/predictions_test_pp/eval_gold.csv" \
    --metrics dice hd95 ndsc
```

Results (76 gold test cases, expert labels):

| Model | VB Dice | Disc Dice | Macro Dice |
|---|---|---|---|
| Dataset002 (gold-trained) | 0.9141 | 0.8623 | 0.8882 |
| Dataset001 (mixed-trained, from `05_model_selection.md`) | 0.9216 | 0.8721 | 0.8969 |

Dataset001 scores marginally higher than Dataset002 on the gold test set despite being trained
on noisier labels — likely because it has ~2.8× more training cases (798 vs 288). This is a
preliminary result for the mixed vs gold comparison; the fairness gap comparison (DPD, DIR)
is what matters for the experiment conclusions.

### Biased ruler (re-aggregated 2026-06-07 — Run 8 in `fairness-runs.md`)

Recomputed Ruler A (Dataset001 vs gold labels) and Ruler B (Dataset001 vs Dataset002 predictions)
on the same 76 gold test images with `dice hd95 ndsc`. Per-case CSVs are written to
`outputs/eval_ruler_gold.csv` and `outputs/eval_ruler_silver.csv` (by `jobs/fairness_analysis.sh`;
the old Run 6 manual run wrote them under `Dataset001_CSpineSeg/predictions_test_pp/`).
The authoritative analysis is the **binarized** rerun in
`outputs/fairness/fairness_biased_ruler/20260607_210826/` (the earlier mean-based Run 6,
`.../biased_ruler/20260605_121654/`, is archived — do not cite its DIR-widening table).

Key result: the generated silver ruler **saturates** (Dice ≈ 0.97 on all 76 cases → zero
failures at threshold 0.8 → DIR ≡ 1.0 for every grouping), so the single-threshold DIR-widening
table is mechanically −100% and uninformative. The biased-ruler signal lives in the continuous
tests instead, and it is **age** — on both rulers. Silver: **11/63 FDR-significant — all age**
(`age_3bin`/`age_median`, Dice & nDSC, 60+ worst). Gold: **0/63 FDR**, but the same-direction age
trend (60+ worst) is its strongest signal too, just sub-FDR (p_fdr≈0.13). So the silver ruler does
not *manufacture* the age effect — it lowers the noise floor: the gap is *larger* against gold
(≈2.7 Dice pts) but noisy, and tiny against silver (≈0.6 pts) but ultra-low-variance (DS001≈DS002
twins), so only the latter clears FDR. Clinically negligible either way. The encoder probe
corroborates age as the salient axis (age decodable above null, race at null; see
`demographic-probing-of-medical-image-encoders/findings.md`). See `fairness-runs.md` Run 8 and
`dpd-dir-redefinition.md` for the full reasoning.

### Dataset003 on gold test set (done 2026-06-10)

Evaluation run via `bias_amplification` stage of `jobs/fairness_analysis.sh` (job 28620483,
~2h40m). Per-case CSV written to `outputs/eval_ds3_on_gold.csv`.

Results (76 gold test cases, expert labels):

| Model | VB Dice | Disc Dice | Macro Dice |
|---|---|---|---|
| Dataset001 (mixed-trained) | 0.9216 | 0.8721 | 0.8969 |
| Dataset002 (gold-trained) | 0.9141 | 0.8623 | 0.8882 |
| **Dataset003 (silver-trained)** | **0.9212** | **0.8721** | **0.8966** |

DS003 scores essentially identically to DS001 and ~0.8 points above DS002 on macro Dice.
The silver-trained model is not degraded by noisy training labels in absolute terms — the
larger training set (450 vs 288) likely compensates.

### CV Validation Dice (not test — scored against silver labels)

| Config | Fold | VB Dice | Disc Dice | Macro |
|---|---|---|---|---|
| 2d | 0 | 0.9796 | 0.9679 | 0.9737 |
| 2d | 1 | 0.9785 | 0.9661 | 0.9723 |
| 2d | 2 | 0.9795 | 0.9687 | 0.9741 |
| 2d | 3 | 0.9779 | 0.9656 | 0.9718 |
| 2d | 4 | 0.9782 | 0.9662 | 0.9722 |
| **2d mean** | | **0.9787** | **0.9669** | **0.9728** |
| 3d_fullres | 0 | 0.9776 | 0.9675 | 0.9725 |
| 3d_fullres | 1 | 0.9762 | 0.9646 | 0.9704 |
| 3d_fullres | 2 | 0.9786 | 0.9678 | 0.9732 |
| 3d_fullres | 3 | 0.9761 | 0.9638 | 0.9700 |
| 3d_fullres | 4 | 0.9758 | 0.9659 | 0.9709 |
| **3d_fullres mean** | | **0.9768** | **0.9659** | **0.9714** |

CV Dice is high because validation is scored against silver labels (same distribution as
training). The gold-label test Dice (0.897) is the meaningful number for comparisons.

### Bias amplification (done 2026-06-10 — Run 9 in `fairness-runs.md`)

Submitted via `sed 's/TPLSTAGE/bias_amplification/g' jobs/fairness_analysis.sh | bsub`.
Results in `outputs/fairness/fairness_bias_amplification/20260609_163752/`.

**0 FDR-significant tests across all three rulers.** Silver-trained DIRs match mixed-trained
almost exactly; gold-trained is occasionally worse on disc fairness (race_wbo disc DIR 0.813
vs silver/mixed 0.875 — only group approaching the 0.80 four-fifths threshold). No evidence
of bias amplification from silver training labels in this dataset.

---

## Design Notes

- **Why 288 gold vs 450 silver train cases?** Structural property of CSpineSeg — fewer cases were expert-annotated. Acknowledged as a limitation; not a choice. The sex-balance (50/50) and race/age stratification are held constant.
- **Same config for all three models** — use whatever `find_best_configuration` selects for Dataset001 for consistency (ensemble 2d + 3d_fullres). This keeps the architecture constant so differences are attributable to training labels, not model capacity.
- **Gold test set is the reference** — all three models are evaluated against the same 76 gold test cases with expert labels. This isolates the effect of training labels on fairness.
