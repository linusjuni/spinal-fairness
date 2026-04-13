# 03 — Training

> **Status: Ready to submit.** Preprocessing and stratified splits are done.

## Pipeline Summary

```
plan_and_preprocess   ✅  done (nnUNetPlannerResEncL, A100 40GB target)
write_splits          ✅  done (splits_final.json with race x age x sex strata)
train 10 jobs         ⬜  submit with: bash jobs/submit.sh
find_best_config      ⬜  after all 10 jobs finish
predict + evaluate    ⬜  after model selection
```

## Planned Configurations

| Config | Patch Size | Batch Size | Notes |
|---|---|---|---|
| `2d` | 512x512 | 12 | Primary — operates on high-res in-plane axes |
| `3d_fullres` | 15x512x512 | — | Anisotropic patch (4mm z-spacing) |

`3d_lowres` was dropped by the planner (volumes too small to warrant it).

## Custom Trainer

`src/nnunet/trainer.py` defines `nnUNetTrainerWandB`, which layers on top of nnU-Net's built-in W&B integration:

- **Built-in** (via `nnUNet_wandb_enabled=1`): logs train/val loss, Dice per class, EMA Dice, learning rate, config, and handles resume
- **Custom trainer adds**: `progress.png` upload each epoch, epoch duration metric, `checkpoint_final.pth` as W&B artifact, `wandb.finish()` cleanup

The trainer file is copied into nnunetv2's package tree at job start (nnU-Net discovers trainers by scanning its own directory, not PYTHONPATH).

## Submitting Jobs

Submit all 10 jobs (5 folds x 2 configs):

```bash
bash jobs/submit.sh
```

This submits each fold as a separate 24h A100 job. All run in parallel.

To submit a single fold for testing:

```bash
sed 's/CONFIG_PLACEHOLDER/2d/g; s/FOLD_PLACEHOLDER/0/g; s/cspine_CONFIG_foldFOLD/cspine_2d_fold0/g; s/CONFIG_fold_FOLD/2d_fold_0/g' jobs/train.sh | bsub
```

Monitor:

```bash
bjobs                 # list jobs
bpeek <JOBID>         # tail live stdout
bnvtop <JOBID>        # GPU utilisation
```

## Resuming

If a job is killed, resubmit the same fold. Add `--c` to the `nnUNetv2_train` command in `train.sh` to resume from the latest checkpoint. W&B resume is automatic.

## Training Outputs

Each fold writes to `$nnUNet_results/Dataset001_CSpineSeg/nnUNetTrainerWandB__nnUNetResEncUNetLPlans__2d/fold_0/`:

```
checkpoint_best.pth       <- saved when EMA pseudo-Dice improves
checkpoint_final.pth      <- saved on completion
progress.png              <- loss + dice curves (also uploaded to W&B)
validation/summary.json   <- per-class Dice on CV validation set
```

## Notes

- First epoch is slow: `torch.compile` JIT compilation + data verification. Don't kill the job.
- `--npz` flag is required for ensembling later. All job scripts include it.
- W&B project: `spinal-fairness`. Login once on a login node with `wandb login`.
