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
   writes preprocessed data to `$nnUNet_preprocessed`.

These can also be run separately: `nnUNetv2_extract_fingerprint`, then
`nnUNetv2_plan_experiment`, then `nnUNetv2_preprocess`.

**After this step, write the custom splits file before training:**
```bash
uv run -m src.nnunet.write_splits
```

### Key flags

| Flag | Purpose |
|---|---|
| `-d DATASET_ID` | Dataset to process (supports multiple) |
| `--verify_dataset_integrity` | Run integrity checks (recommended on first run) |
| `-pl PLANNER_CLASS` | Planner to use (default: `ExperimentPlanner`; see ResEnc planners below) |
| `-gpu_memory_target N` | Override default 8 GB VRAM target (useful on A100s) |
| `-overwrite_target_spacing X Y Z` | Override target spacing (creates custom plans) |
| `-overwrite_plans_name NAME` | Custom plans file name (avoids overwriting default) |

### Expected configuration output for CSpineSeg

The `ANISO_THRESHOLD` in nnU-Net is **3**. Our ~8x anisotropy (~0.5 mm in-plane, ~4 mm
through-plane) far exceeds this. This triggers special handling:

- **Target z-spacing** is set to the **10th percentile** of observed z-spacings (~3.3-3.6 mm)
  rather than the median, preventing excessive interpolation in the low-resolution axis.
- **Resampling** uses 3rd-order spline in-plane but **nearest neighbor along z** to avoid
  interpolation artifacts between sparse slices.
- **Axes are transposed** so the lowest-resolution axis (z) comes first.
- **Pooling** is performed exclusively in-plane first until resolution between axes matches,
  then 3D pooling begins. Convolution kernels use size 1 along z initially.

| Configuration | Likelihood | Notes |
|---|---|---|
| `2d` | **Very likely primary** | Always generated. Operates on the two high-resolution in-plane axes. |
| `3d_fullres` | Likely generated | With only ~4M voxels per volume, the patch can cover most of the image. Patch will be anisotropic (e.g., ~[16, 256, 256]). |
| `3d_lowres` | Unlikely | Only generated if 3d_fullres patch covers <25% of median volume. Our small volumes make this unlikely. |
| `3d_cascade_fullres` | Only if 3d_lowres exists | Unlikely |

Check `$nnUNet_preprocessed/Dataset001_CSpineSeg/nnUNetPlans.json` after planning to see
exactly what was decided.

> **Caveat:** Do not manually edit `nnUNetPlans.json` unless you know what you are doing.
> If you want to override target spacing, use `-overwrite_target_spacing` and
> `-overwrite_plans_name` to create a differently-named plans file rather than
> overwriting the default.

### Published Baseline (CSpineSeg paper)

Zhai et al. (*Scientific Data*, 2025) trained nnU-Net on this exact dataset and reported:

| Configuration | Vertebral Body Dice | Disc Dice | Macro-Average |
|---|---|---|---|
| 2D | reported | reported | reported |
| 3D fullres | reported | reported | reported |
| **Ensemble** | **0.929** | **0.904** | **0.916** |

Differences between 2D, 3D, and ensemble were **not statistically significant** (P > 0.05).
Their code is at [github.com/JikaiZ/CSpineSeg](https://github.com/JikaiZ/CSpineSeg).

---

## Residual Encoder Planners (v2.4.1+)

nnU-Net v2.4.1 introduced Residual Encoder UNets with dedicated planners that target
different VRAM budgets. On an A100 these are worth considering:

| Planner | Target VRAM | Notes |
|---|---|---|
| `ExperimentPlanner` (default) | 8 GB | Standard planner, works on any GPU >= 11 GB |
| `nnUNetPlannerResEncM` | 9-11 GB | Medium ResEnc; removes 5% batch size cap |
| `nnUNetPlannerResEncL` | 24 GB | Large ResEnc; good for A100 40 GB |
| `nnUNetPlannerResEncXL` | 40 GB | Extra-large ResEnc; for A100 80 GB |

Usage:
```bash
nnUNetv2_plan_and_preprocess -d 1 -pl nnUNetPlannerResEncL
nnUNetv2_train 1 3d_fullres 0 --npz -p nnUNetResEncUNetLPlans
```

The plans file name changes with the planner, so the `-p` flag at training time must match.

---

## Step 2: Train All Configurations

Train all 5 folds for each configuration. The `--npz` flag saves softmax outputs needed
for ensembling in `find_best_configuration`.

```bash
# 2D configuration — all 5 folds
for fold in 0 1 2 3 4; do
    nnUNetv2_train 1 2d $fold --npz
done

# 3D full-res configuration — all 5 folds
for fold in 0 1 2 3 4; do
    nnUNetv2_train 1 3d_fullres $fold --npz
done
```

On the cluster, each fold is a separate job (see LSF scripts below).

### Important training flags

| Flag | Purpose |
|---|---|
| `--npz` | Save softmax predictions during final validation. **Required for ensembling** in `find_best_configuration`. If omitted, you can retroactively generate them: `nnUNetv2_train 1 2d 0 --val --npz` (runs validation only). |
| `-tr TRAINER` | Custom trainer class name. See [04 — Experiment Tracking](04_experiment_tracking.md) for details. |
| `--c` | Resume from latest checkpoint. Searches in order: `checkpoint_final.pth` -> `checkpoint_latest.pth` -> `checkpoint_best.pth`. |
| `-num_gpus N` | Multi-GPU DDP training. See note below. |
| `-device DEVICE` | `cuda`, `cpu`, or `mps`. |
| `--val` | Run validation only (requires finished training). Useful with `--npz` to retroactively generate softmax outputs. |
| `-pretrained_weights PATH` | Path to `.pth` checkpoint for weight initialization. |

> **Note on `-num_gpus`:** nnU-Net supports multi-GPU DDP training but the developers
> explicitly recommend running separate folds on separate GPUs instead. DDP has
> communication overhead and provides diminishing returns for the small batch sizes
> typical in nnU-Net (2-12). On an A100 node with 2 GPUs, run two folds in parallel.

### Resuming interrupted training

```bash
nnUNetv2_train 1 2d 0 --npz --c
```

The `--c` flag resumes from the latest available checkpoint. nnU-Net saves a checkpoint
every 50 epochs by default.

---

## `torch.compile` (Default On)

Since ~v2.5, `torch.compile` is **enabled by default** on CUDA (controlled by the
`nnUNet_compile` environment variable). This compiles both the network and the Dice loss
component.

**Implications:**
- First epoch is significantly slower due to JIT compilation. Do not kill the job if GPU
  utilisation is 0% for the first several minutes.
- Subsequent epochs are faster.
- To disable: `export nnUNet_compile=false` in your job script.
- Known issues: not supported on Windows (no Triton wheels); crash bug with CE loss on
  torch 2.2.2.

---

## A100-Specific Notes

DTU HPC has two A100 variants in the `gpua100` queue:
- **4 nodes** — Tesla A100 PCIe **40 GB**
- **6 nodes** — Tesla A100 PCIe **80 GB**

Request a specific variant by adding a resource requirement (see job scripts below).

### Why the A100 matters for nnU-Net

| Property | Impact |
|---|---|
| 40-80 GB VRAM | Enables ResEnc planners (`nnUNetPlannerResEncL/XL`) with larger patch sizes and deeper networks. The default 8 GB planner does not fully utilise A100 capacity. |
| BFloat16 (BF16) support | Ampere architecture supports BF16 natively. BF16 has the same dynamic range as FP32 (unlike FP16) with half the memory. See BF16 section below. |
| Tensor Cores (3rd gen) | Matrix multiplications are accelerated. Mixed-precision training on A100 is 1.3-2.5x faster than on V100. |

### Enabling BF16 in a custom trainer

nnU-Net v2 uses `torch.autocast` with **FP16 by default** — the autocast call does not
pass a `dtype` argument. There is **no `self.autocast_dtype` attribute** on the base trainer.

To use BF16, you must **override `train_step()` and `validation_step()`** to pass
`dtype=torch.bfloat16` to the `autocast()` context manager:

```python
from torch.amp import autocast

class nnUNetTrainerBF16(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Disable GradScaler — BF16 does not need loss scaling
        # (GradScaler is a no-op with BF16 since PyTorch 2.3+, but
        # disabling it avoids unnecessary overhead)
        self.grad_scaler = None

    def train_step(self, batch):
        # Identical to parent but with dtype=torch.bfloat16
        data = batch['data']
        target = batch['target']
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True, dtype=torch.bfloat16):
            output = self.network(data)
            l = self.loss(output, target)
        l.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
```

> **Note:** Since PyTorch 2.3+, `GradScaler` is officially a no-op for dtypes that do not
> need loss scaling (like BF16). Leaving it in is safe but unnecessary. Setting
> `self.grad_scaler = None` causes the base trainer's code to take the non-scaled branch
> (`l.backward()` directly).

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

# Optional: enable built-in W&B logging
export nnUNet_wandb_enabled=1
export nnUNet_wandb_project="spinal-fairness"

nnUNetv2_train 1 2d 0 --npz
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

> **Caveat:** The first epoch is slow for two reasons: (1) nnU-Net extracts/verifies
> preprocessed data on first access, and (2) `torch.compile` (enabled by default) performs
> JIT compilation. Do not kill the job if GPU utilisation is 0% for the first several
> minutes. Only start the next fold **after** confirming GPU utilisation is non-zero on fold 0.

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
├── checkpoint_best.pth       <- saved when EMA pseudo-Dice improves
├── checkpoint_latest.pth     <- saved every 50 epochs (deleted at end of training)
├── checkpoint_final.pth      <- saved on training completion
├── progress.png              <- 3-panel plot: loss+dice, epoch duration, LR schedule
├── training_log_*.txt        <- per-epoch text log
├── debug.json                <- device, PyTorch version, GPU name, data transforms
├── plans.json                <- copy of plans file
├── dataset.json              <- copy of dataset descriptor
├── dataset_fingerprint.json  <- copy of fingerprint
└── validation/               <- predicted segmentations for validation cases
    ├── summary.json           <- per-class Dice metrics
    └── *.npz                  <- softmax files (if --npz was used)
```

The `progress.png` is regenerated every epoch and shows losses, pseudo-Dice, EMA Dice,
epoch duration, and learning rate. It is also uploaded to W&B if the built-in W&B
integration is enabled (see [04 — Experiment Tracking](04_experiment_tracking.md)).

At the end of training, `checkpoint_latest.pth` is deleted. Only `checkpoint_best.pth`
and `checkpoint_final.pth` remain.
