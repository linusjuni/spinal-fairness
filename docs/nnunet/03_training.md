# 03 — Training

## Step 1: Plan and Preprocess

```bash
nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity
```

This command:
1. **Extracts a dataset fingerprint** — reads all image spacings, shapes, and intensity
   statistics from the raw NIfTI files.
2. **Runs the experiment planner** — determines optimal configurations (patch size, batch
   size, architecture depth) for each of `2d`, `3d_fullres`, and `3d_lowres`.
3. **Preprocesses the data** — resamples to target spacing, normalises intensities,
   writes preprocessed `.npz` files to `$nnUNet_preprocessed`.

**After this step, write the custom splits file before training:**
```bash
uv run -m src.nnunet.write_splits
```

### Expected configuration output for CSpineSeg

Given our ~8× anisotropy (~0.5 mm in-plane, ~4 mm through-plane):

| Configuration | Likelihood | Notes |
|---|---|---|
| `2d` | **Very likely primary** | nnU-Net defaults to 2D for highly anisotropic data |
| `3d_fullres` | Likely generated | A100 has enough VRAM; patch size will be anisotropic |
| `3d_lowres` | Only if 3d_fullres patch < 1/8 median volume | Unlikely with our data size |
| `3d_cascade_fullres` | Only if 3d_lowres exists | Unlikely |

Check `$nnUNet_preprocessed/Dataset001_CSpineSeg/nnUNetPlans.json` after planning to see
exactly what was decided.

> **Caveat:** Do not manually edit `nnUNetPlans.json` unless you know what you are doing.
> If you want to override target spacing, use `-overwrite_target_spacing` and
> `-overwrite_plans_name` to create a differently-named plans file rather than
> overwriting the default.

---

## Step 2: Train All Configurations

Train all 5 folds for each configuration. The `--npz` flag saves softmax outputs needed
for `find_best_configuration` — **always include it**.

```bash
# 2D configuration — all 5 folds
for fold in 0 1 2 3 4; do
    nnUNetv2_train 1 2d $fold --npz -tr nnUNetTrainerWandB
done

# 3D full-res configuration — all 5 folds
for fold in 0 1 2 3 4; do
    nnUNetv2_train 1 3d_fullres $fold --npz -tr nnUNetTrainerWandB
done
```

On the cluster, each fold is a separate job (see LSF scripts below).

> **Caveat on `-num_gpus`:** nnU-Net supports multi-GPU DDP training but the developers
> explicitly note it is **slower** than running separate folds on separate GPUs. On an
> A100 node with 2 GPUs, run two folds in parallel rather than one fold with both GPUs.

### Resuming interrupted training

```bash
nnUNetv2_train 1 2d 0 --npz -tr nnUNetTrainerWandB --c
```

The `--c` flag resumes from `checkpoint_latest.pth`. nnU-Net saves a checkpoint every
50 epochs by default.

---

## A100-Specific Notes

DTU HPC has two A100 variants in the `gpua100` queue:
- **4 nodes** — Tesla A100 PCIe **40 GB**
- **6 nodes** — Tesla A100 PCIe **80 GB**

Request a specific variant by adding a resource requirement (see job scripts below).

### Why the A100 matters for nnU-Net

| Property | Impact |
|---|---|
| 40–80 GB VRAM | nnU-Net `3d_fullres` can use large patch sizes that would OOM on smaller GPUs (e.g. RTX 3090 at 24 GB). With CSpineSeg's moderate volume size this is unlikely to be the bottleneck, but it removes the constraint entirely. |
| BFloat16 (BF16) support | Ampere architecture supports BF16 natively. BF16 has the same dynamic range as FP32 (unlike FP16) with half the memory. nnU-Net v2 uses PyTorch AMP which will use FP16 by default — BF16 is more numerically stable on A100. |
| Tensor Cores (3rd gen) | Matrix multiplications are accelerated. Mixed-precision training on A100 is 1.3–2.5× faster than on V100. |
| NVLink (SXM variants) | Not applicable here — PCIe variants. Multi-GPU communication via PCIe is slower. |

### Enabling BF16 in the custom trainer

In `nnUNetTrainerWandB`, add to `__init__`:
```python
# Prefer BF16 on Ampere GPUs for stability
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    self.autocast_dtype = torch.bfloat16
```

> **Caveat:** nnU-Net v2's default `GradScaler` is designed for FP16. BF16 does not need
> gradient scaling (it doesn't underflow like FP16). If you enable BF16, disable the
> scaler in your trainer subclass to avoid unnecessary overhead.

---

## DTU HPC — LSF Job Scripts

### Single fold job (recommended pattern)

Save as `jobs/train_fold.sh` and submit with `bsub < jobs/train_fold.sh`:

```bash
#!/bin/bash
#BSUB -J cspine_train_2d_fold0
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 4
#BSUB -W 24:00
#BSUB -o logs/train_2d_fold0_%J.out
#BSUB -e logs/train_2d_fold0_%J.err

# Environment
source ~/.bashrc
source .venv/bin/activate

export nnUNet_raw="/work3/s225224/nnunet/raw"
export nnUNet_preprocessed="/work3/s225224/nnunet/preprocessed"
export nnUNet_results="/work3/s225224/nnunet/results"

nnUNetv2_train 1 2d 0 --npz -tr nnUNetTrainerWandB
```

### Submitting all folds at once

```bash
for config in 2d 3d_fullres; do
    for fold in 0 1 2 3 4; do
        sed "s/FOLD/$fold/g; s/CONFIG/$config/g" jobs/train_template.sh \
            | bsub
    done
done
```

### Resource guidelines

| Resource | Value | Rationale |
|---|---|---|
| GPUs | 1 per fold | Multi-GPU DDP is slower for nnU-Net |
| CPUs (`-n`) | 4 | nnU-Net data augmentation uses multiprocessing |
| RAM per core | 8 GB | Covers preprocessing buffers; increase to 16 GB if OOM on CPU |
| Wall time | 24h | A single 2D fold on CSpineSeg (~1,000 cases, 1,000 epochs) |
| Queue | `gpua100` | Targets A100 nodes |

> **Caveat:** nnU-Net's first epoch is slow — it extracts/verifies preprocessed data on
> first access. Do not kill the job if GPU utilisation is 0% for the first few minutes.
> Only start the next fold **after** confirming GPU utilisation is non-zero on fold 0.

### Monitoring

```bash
bjobs -l <JOBID>      # detailed job info
bpeek <JOBID>         # tail live stdout
bnvtop <JOBID>        # GPU utilisation (interactive)
```

---

## Training Outputs

Each fold writes to:
```
$nnUNet_results/Dataset001_CSpineSeg/nnUNetTrainer__nnUNetPlans__2d/fold_0/
├── checkpoint_best.pth       ← saved when EMA pseudo-Dice improves
├── checkpoint_latest.pth     ← saved every 50 epochs (deleted at end)
├── checkpoint_final.pth      ← saved on training completion
├── progress.png              ← live training curves (loss, dice, LR)
└── training_log_*.txt        ← per-epoch text log
```

The `progress.png` is regenerated every epoch and shows losses, pseudo-Dice, and
learning rate. It is also uploaded to W&B by the custom trainer (see
[04 — Experiment Tracking](04_experiment_tracking.md)).
