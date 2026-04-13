# 03 — Training

> **Status: Running.** 10 jobs submitted to `gpua100` on 2026-04-13.

## Pipeline Summary

```
plan_and_preprocess   ✅  done (nnUNetPlannerResEncL)
write_splits          ✅  done (splits_final.json, 914 cases, 5-fold stratified by race x age x sex)
train 10 jobs         ⏳  running — 5 folds x 2d + 3d_fullres, job IDs 28200038–28200047
find_best_config      ⬜  after all 10 jobs finish
predict + evaluate    ⬜  after model selection
```

## Configurations

| Config | Patch Size | Batch Size | Epoch time (A100) | Wall time |
|---|---|---|---|---|
| `2d` | 512×512 | 35 | ~3 min | 72h (covers ~1000 epochs) |
| `3d_fullres` | 15×512×512 | 2 | ~6–10 min (est.) | 72h (partial — needs resume) |

`3d_lowres` was dropped by the planner (volumes too small).

## Custom Trainer

`src/nnunet/trainer.py` → `nnUNetTrainerWandB`:

- **Built-in** (`nnUNet_wandb_enabled=1`): loss, Dice, EMA Dice, LR, resume
- **Custom adds**: `progress.png` per epoch, epoch duration, `checkpoint_final.pth` artifact, `wandb.finish()`

Copied into nnunetv2's package tree at job start (nnU-Net discovers trainers by scanning its own tree, not PYTHONPATH).

## Submitting Jobs

```bash
bash jobs/submit.sh    # all 10 jobs
```

Single fold:

```bash
sed 's/TPLCONFIG/2d/g; s/TPLFOLD/0/g' jobs/train.sh | bsub
```

Monitor:

```bash
bjobs                  # list jobs (NALLOC values for pending jobs are a display artifact — ignore)
bpeek <JOBID>          # tail live stdout
bnvtop <JOBID>         # GPU utilisation
```

## Resuming

If a job is killed before completion, resubmit the same fold with `--c` added to the `nnUNetv2_train` call in `train.sh`. W&B resume is automatic (detects existing `wandb/` directory).

3d_fullres will need at least one resume (72h ≈ 430–720 epochs at 6–10 min/epoch).

## Training Outputs

```
$nnUNet_results/Dataset001_CSpineSeg/nnUNetTrainerWandB__nnUNetResEncUNetLPlans__2d/fold_N/
├── checkpoint_best.pth       <- saved when EMA pseudo-Dice improves
├── checkpoint_final.pth      <- saved on completion
├── progress.png              <- loss + Dice curves (also in W&B)
└── validation/summary.json   <- per-class Dice on CV validation set
```

## Notes

- **torch.compile cold cache**: first epoch takes 30–45 min on a fresh node while Triton compiles kernels. GPU shows 0% util but `.err` will have inductor warnings — this is normal. Subsequent epochs ~3 min.
- **`--npz` required** for ensembling in `find_best_configuration`. All job scripts include it.
- **W&B**: project `spinal-fairness`, run names `2d_fold0` etc. set by `on_train_start` hook. Login once: `uv run wandb login`.
- **Quota**: at ~128 GiB / 300 GiB on work3. Training results add ~10–30 GB.
