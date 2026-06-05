# 07 — Hugging Face Release Plan

> **Status (2026-06-05):** Plan only — nothing uploaded yet. Dataset001 (mixed) and
> Dataset002 (gold) are fully trained and exportable. Dataset003 (silver) is 7/10
> (3d_fullres folds 0/1/4 pending) — ship it once those finish, or release 001+002 first.

Goal: get the trained nnU-Net models off the DTU HPC and onto Hugging Face so others can
install and run inference, with enough documentation that an outside user can reproduce the
reported numbers.

Everything in **our** workflow runs through `uv` (export, upload). The end-user inference
snippet in the model card stays generic (`pip`) because external users won't have this
project's venv.

## Path overview

```
HPC results dir ──(re-stamp trainer)──► uv run nnUNetv2_export_model_to_zip ──► .zip per dataset
       │                                                                          │
       └──────────────── uv run hf upload (login node, CPU, internet) ───────────┘
                                         │
                                         ▼
                          HF repo (private) + model card (README.md)
                                         │
     user: hf download ─► nnUNetv2_install_pretrained_model_from_zip ─► nnUNetv2_predict
```

All HPC steps run on a **login node**: export is CPU-only and the upload needs internet
(compute nodes are firewalled). No GPU required.

---

## Pre-flight

1. **Licensing gate.** We may upload model *weights* but **not** the CSpineSeg data/labels
   (MIDRC DUA prohibits redistributing the data; silver labels are also derivatives — keep
   them off too). Keep the repo **private** until there is written permission from the
   CSpineSeg corresponding author (jikai.zhang@duke.edu) for a public derived-model release.
   License the model **non-commercial** (CC-BY-NC-4.0) to match the MIDRC DUA track.
   See the DUA: <https://www.midrc.org/midrc-data-use-agreement>.

2. **What's ready.** Dataset001 ✅ and Dataset002 ✅ fully trained. Dataset003 = 7/10
   (3d_fullres folds 0/1/4 pending, see `06_gold_silver_training.md`).

3. **Verify the custom trainer is inference-neutral.** Open the `nnUNetTrainerWandB` source
   and confirm it only adds W&B logging and does **not** override `build_network_architecture`.
   This is what makes the re-stamp step (B-1) safe. If it *does* override the network builder,
   use the shim route (B-2 alternative) instead.

---

## Step A — Repo shape

One HF repo for the project (a model *family*), one zip per dataset, one model card
describing all three:

```
spinal-fairness-cspineseg/            (private model repo)
├── README.md                         ← model card (see Step D)
├── Dataset001_CSpineSeg_mixed.zip
├── Dataset002_CSpineSeg_gold.zip
├── Dataset003_CSpineSeg_silver.zip   (when ready)
├── postprocessing/
│   ├── Dataset001_ensemble_postprocessing.pkl
│   ├── Dataset001_ensemble_plans.json
│   └── ...                           (per dataset)
└── nnUNetTrainerWandB.py             ← only if using the shim route (B-2 alt)
```

---

## Step B — Export each model to a zip (HPC login node)

### B-1. Re-stamp the trainer name (recommended)

The checkpoint stores `trainer_name='nnUNetTrainerWandB'`, and `nnUNetv2_predict` tries to
import that class. Rather than make every user install our trainer, copy the results to a
staging dir and rewrite the name to the stock `nnUNetTrainer`. Run once:

```bash
cd /work3/s225224/spinal-fairness     # this repo on the HPC
export STAGE=/work3/s225224/nnunet/export_stage
source .env                            # sets $nnUNet_results

uv run python - <<'EOF'
import os, shutil, torch
from pathlib import Path

src_results = Path(os.environ["nnUNet_results"])
stage       = Path(os.environ["STAGE"]); stage.mkdir(parents=True, exist_ok=True)

OLD, NEW = "nnUNetTrainerWandB", "nnUNetTrainer"
for ds in ["Dataset001_CSpineSeg", "Dataset002_CSpineSeg_Gold", "Dataset003_CSpineSeg_Silver"]:
    s = src_results / ds
    if not s.exists():
        print(f"skip {ds} (not found)"); continue
    d = stage / ds
    if d.exists(): shutil.rmtree(d)
    shutil.copytree(s, d, symlinks=False)              # resolves symlinked checkpoints
    for sub in list(d.iterdir()):
        if sub.is_dir() and sub.name.startswith(OLD + "__"):
            new_dir = sub.with_name(sub.name.replace(OLD, NEW, 1))
            sub.rename(new_dir)
            for ckpt in new_dir.rglob("checkpoint_final.pth"):
                obj = torch.load(ckpt, map_location="cpu", weights_only=False)
                if obj.get("trainer_name") == OLD:
                    obj["trainer_name"] = NEW
                    torch.save(obj, ckpt)
            print(f"restamped {new_dir.name}")
EOF
```

### B-2. Export to zip

```bash
nnUNet_results=$STAGE uv run nnUNetv2_export_model_to_zip \
  -d 1 \
  -o Dataset001_CSpineSeg_mixed.zip \
  -c 2d 3d_fullres \
  -tr nnUNetTrainer \
  -p nnUNetResEncUNetLPlans \
  -f 0 1 2 3 4
# repeat: -d 2 -o Dataset002_CSpineSeg_gold.zip
#         -d 3 -o Dataset003_CSpineSeg_silver.zip   (add --not_strict if shipping <10 folds)
```

Flags (from nnU-Net source `nnunetv2/model_sharing/entry_points.py`):

| Flag | Meaning | Default |
|---|---|---|
| `-d` | dataset name/id (**required**) | — |
| `-o` | output zip path (**required**) | — |
| `-c` | configurations | all four |
| `-tr` | trainer class | `nnUNetTrainer` |
| `-p` | plans identifier | `nnUNetPlans` |
| `-f` | folds | `0 1 2 3 4` |
| `-chk` | checkpoint name | `checkpoint_final.pth` |
| `--not_strict` | allow missing folds/configs | off |
| `--exp_cv_preds` | also bundle CV predictions | off |

The zip bundles, per fold: network weights + `plans.json` + `dataset.json`.

**B-2 alternative — shim route** (use only if the trainer overrides the network builder):
skip B-1, export with `-tr nnUNetTrainerWandB`, and ship a one-file shim
`nnUNetTrainerWandB.py`:

```python
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
class nnUNetTrainerWandB(nnUNetTrainer):
    pass
```

Document that users drop it into `nnunetv2/training/nnUNetTrainer/` before predicting and
pass `-tr nnUNetTrainerWandB`.

### B-3. Grab the ensemble + postprocessing artifacts

`export_model_to_zip` does **not** include the ensemble postprocessing, but the reported
numbers are the 2d+3d ensemble *after* postprocessing. Copy those out so users can
reproduce them (Dataset001 shown; repeat per dataset):

```bash
ENS=$nnUNet_results/Dataset001_CSpineSeg/ensembles/ensemble___nnUNetTrainerWandB__nnUNetResEncUNetLPlans__2d___nnUNetTrainerWandB__nnUNetResEncUNetLPlans__3d_fullres___0_1_2_3_4
mkdir -p postprocessing
cp $ENS/postprocessing.pkl postprocessing/Dataset001_ensemble_postprocessing.pkl
cp $ENS/plans.json         postprocessing/Dataset001_ensemble_plans.json
```

---

## Step C — Upload to Hugging Face (HPC login node)

CLI is `hf` (old `huggingface-cli` name still works). Use a **write** token from
<https://huggingface.co/settings/tokens>.

```bash
uv run hf auth login                  # paste write token; or: export HF_TOKEN=hf_xxx

uv run hf repo create spinal-fairness-cspineseg --repo-type model --private

# large multi-GB zips, resume-on-drop (best over HPC links):
uv run hf upload-large-folder <username>/spinal-fairness-cspineseg . --repo-type model

# or single files:
uv run hf upload <username>/spinal-fairness-cspineseg \
  Dataset001_CSpineSeg_mixed.zip --repo-type model
```

`upload-large-folder` chunks, dedups, and **resumes** if the connection drops. Total
footprint (3 datasets × 2 configs × 5 folds of ResEncUNet-L) is well under the free tier's
**100 GB private** quota.

---

## Step D — Model card (`README.md`) contents

This is the user-facing documentation. Required pieces:

**1. YAML frontmatter:**
```yaml
---
license: cc-by-nc-4.0
library_name: nnunetv2
pipeline_tag: image-segmentation
tags: [nnunet, medical-imaging, mri, spine, segmentation, fairness]
---
```

**2. Model description** — nnU-Net v2, `nnUNetResEncUNetLPlans` (ResEnc-L), configs `2d` +
`3d_fullres` + their ensemble; task = cervical-spine **vertebral body** + **intervertebral
disc** segmentation on **sagittal T2 MRI**.

**3. Three-model table** (from `06_gold_silver_training.md`): Dataset001 (mixed,
deploy-realistic), 002 (gold/expert), 003 (silver/auto). Users must know which ruler each
was trained on.

**4. Input spec** (the #1 source of user error):
- Single channel, sagittal **T2-weighted** MRI, NIfTI.
- nnU-Net naming: each case file ends `_0000.nii.gz` (channel 0000).
- Output labels: `0`=background, `1`=vertebral body, `2`=disc.
- nnU-Net resamples/normalizes internally from `plans.json` — do **not** pre-normalize.

**5. Performance tables** — CV Dice + test Dice (gold/silver/all) from `05_model_selection.md`,
plus the Zhou et al. baseline comparison.

**6. Fairness caveats** (essential for this project): CSpineSeg demographics (45/55 M/F;
65% White / 28% Black / 2% Asian; age 55±17), the gold-vs-silver "biased ruler" gap (same
predictions score ~8 Dice points higher against auto-labels), and an explicit statement
that per-demographic performance is **not validated → research use only, not for clinical use**.

**7. Install + inference commands** — see Step E.

**8. nnU-Net version pin** — "tested with nnunetv2 ≥ 2.x; install torch first; avoid
torch 2.9.* (3D-conv AMP regression)" (from `01_setup.md`).

**9. Attribution / citation** (mandatory under the DUA): Zhou et al. *Scientific Data* 2025
(`10.1038/s41597-025-05975-w`), acknowledge MIDRC (`10.60701/H6K0-A61V`), plus our own
thesis/paper.

**10. License & contact** — CC-BY-NC-4.0; Linus Juni; supervisors Aditya Parikh, Aasa Feragen.

---

## Step E — What an end user runs (put verbatim in the card)

End users won't have our `uv` project, so the card uses plain `pip`:

```bash
# 0. install nnU-Net v2 (torch first)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install nnunetv2 "huggingface_hub[cli]"

# 1. set the three required env vars
export nnUNet_raw=/path/raw nnUNet_preprocessed=/path/pre nnUNet_results=/path/results

# 2. download + install the model into $nnUNet_results
hf download <username>/spinal-fairness-cspineseg Dataset001_CSpineSeg_mixed.zip --local-dir .
nnUNetv2_install_pretrained_model_from_zip Dataset001_CSpineSeg_mixed.zip

# 3. predict each config, then ensemble + postprocess (reproduces reported numbers)
nnUNetv2_predict -i INPUT -o OUT_2d -d 1 -c 2d         -p nnUNetResEncUNetLPlans -f 0 1 2 3 4 --save_probabilities
nnUNetv2_predict -i INPUT -o OUT_3d -d 1 -c 3d_fullres -p nnUNetResEncUNetLPlans -f 0 1 2 3 4 --save_probabilities
nnUNetv2_ensemble -i OUT_2d OUT_3d -o OUT_ens -np 8
nnUNetv2_apply_postprocessing -i OUT_ens -o OUT_final \
  -pp_pkl_file postprocessing/Dataset001_ensemble_postprocessing.pkl \
  -plans_json  postprocessing/Dataset001_ensemble_plans.json
```

If the re-stamp route (B-1) was used, this runs on **stock** nnU-Net with no extra files.
If the shim route was used, add: place `nnUNetTrainerWandB.py` in
`nnunetv2/training/nnUNetTrainer/` and use `-tr nnUNetTrainerWandB`.

---

## Checklist

- [ ] Written OK from CSpineSeg author for public release (else stay **private**)
- [ ] Confirmed `nnUNetTrainerWandB` doesn't override the network builder → B-1 safe
- [ ] Exported 001, 002 (003 when folds 0/1/4 finish — `--not_strict` if partial)
- [ ] Copied ensemble `postprocessing.pkl` + `plans.json` per dataset
- [ ] Repo created **private**, uploaded with `upload-large-folder`
- [ ] Model card: input spec, label map, Dice tables, **fairness caveats**, MIDRC+Zhou
      citation, NC license, version pin

---

## References

- nnU-Net export/install: <https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md>
- Export flags source: `nnunetv2/model_sharing/entry_points.py`
- HF CLI: <https://huggingface.co/docs/huggingface_hub/guides/cli>
- HF storage limits (100 GB private free tier): <https://huggingface.co/docs/hub/storage-limits>
- Example nnU-Net v2 model card: <https://huggingface.co/ruiruili/LUMEN_CamSVD_nnUNet>
- MIDRC DUA: <https://www.midrc.org/midrc-data-use-agreement>
- CSpineSeg data descriptor: Zhou et al., *Scientific Data* 2025, doi:10.1038/s41597-025-05975-w
