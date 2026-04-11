# 04 — Experiment Tracking

## Overview

nnU-Net v2 has no built-in W&B support. The right integration point is a **custom trainer
subclass** that overrides the logging hooks in `nnUNetTrainer`. This keeps our changes
fully contained — no patches to the nnU-Net package are needed.

The trainer is selected at training time via the `-tr` flag:
```bash
nnUNetv2_train 1 2d 0 --npz -tr nnUNetTrainerWandB
```

nnU-Net discovers the trainer class by importing it from `PYTHONPATH`. As long as our
`src/` is on the path (which it is when running via `uv run -m`), no registration step
is needed.

---

## What nnUNetTrainer Logs Internally

Understanding the internal logging hooks is necessary to know what to intercept:

| Hook method | Metrics available | When called |
|---|---|---|
| `on_train_epoch_end` | `train_losses` | After each training epoch |
| `on_validation_epoch_end` | `val_losses`, `mean_fg_dice`, `dice_per_class_or_region` | After each validation epoch |
| `on_epoch_end` | `lrs`, `ema_fg_dice`, `epoch_*_timestamps` | End of each full epoch; also saves `progress.png` |
| `on_train_end` | — | Training complete; saves `checkpoint_final.pth` |

Metrics are stored on `self.logger` (a `MetaLogger` instance). Retrieve the latest value
with `self.logger.my_fantastic_logging_values[-1]` or via internal list indexing.

---

## nnUNetTrainerWandB Implementation

Location: `src/nnunet/trainer.py`

```python
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import wandb
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

if TYPE_CHECKING:
    pass


class nnUNetTrainerWandB(nnUNetTrainer):
    """nnUNetTrainer subclass that logs metrics and artifacts to Weights & Biases."""

    def __init__(self, plans, configuration, fold, dataset_json, unpack_dataset=True,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # Prefer BF16 on Ampere GPUs (A100) for numerical stability
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            self.autocast_dtype = torch.bfloat16

        if self.local_rank == 0:  # Only rank-0 process initialises W&B
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "spinal-fairness"),
                name=f"{configuration}_fold{fold}",
                group=configuration,
                tags=[configuration, f"fold{fold}", "nnunet"],
                config={
                    "dataset": "CSpineSeg",
                    "configuration": configuration,
                    "fold": fold,
                    "plans": plans,
                },
                resume="allow",  # Safe to re-run after a crash
            )

    def on_train_epoch_end(self, train_outputs: list[dict]) -> None:
        super().on_train_epoch_end(train_outputs)
        if self.local_rank == 0:
            wandb.log(
                {"train/loss": self.logger.my_fantastic_logging_values["train_losses"][-1]},
                step=self.current_epoch,
            )

    def on_validation_epoch_end(self, val_outputs: list[dict]) -> None:
        super().on_validation_epoch_end(val_outputs)
        if self.local_rank == 0:
            dice_per_class = self.logger.my_fantastic_logging_values[
                "dice_per_class_or_region"
            ][-1]
            log_dict = {
                "val/loss": self.logger.my_fantastic_logging_values["val_losses"][-1],
                "val/mean_fg_dice": self.logger.my_fantastic_logging_values[
                    "mean_fg_dice"
                ][-1],
                "val/dice_vertebral_body": dice_per_class[0]
                if len(dice_per_class) > 0
                else None,
                "val/dice_disc": dice_per_class[1] if len(dice_per_class) > 1 else None,
            }
            wandb.log({k: v for k, v in log_dict.items() if v is not None},
                      step=self.current_epoch)

    def on_epoch_end(self) -> None:
        super().on_epoch_end()
        if self.local_rank == 0:
            wandb.log(
                {
                    "train/lr": self.logger.my_fantastic_logging_values["lrs"][-1],
                    "val/ema_fg_dice": self.logger.my_fantastic_logging_values[
                        "ema_fg_dice"
                    ][-1],
                },
                step=self.current_epoch,
            )
            # Upload the live progress plot as a W&B image
            progress_path = self.output_folder / "progress.png"
            if progress_path.exists():
                wandb.log(
                    {"charts/progress": wandb.Image(str(progress_path))},
                    step=self.current_epoch,
                )

    def on_train_end(self) -> None:
        super().on_train_end()
        if self.local_rank == 0:
            # Log the final checkpoint as a W&B artifact for reproducibility
            artifact = wandb.Artifact(
                name=f"checkpoint-fold{self.fold}",
                type="model",
                description=f"Final nnU-Net checkpoint, fold {self.fold}",
            )
            checkpoint_path = self.output_folder / "checkpoint_final.pth"
            if checkpoint_path.exists():
                artifact.add_file(str(checkpoint_path))
                wandb.log_artifact(artifact)
            wandb.finish()
```

---

## W&B Run Structure

Each fold produces one W&B run. Grouping by configuration lets you compare 2D vs 3D
across folds in the same W&B project view.

| W&B field | Value |
|---|---|
| Project | `spinal-fairness` (override via `WANDB_PROJECT` env var) |
| Run name | `2d_fold0`, `2d_fold1`, ..., `3d_fullres_fold0`, ... |
| Group | `2d` or `3d_fullres` |
| Tags | configuration + fold + `nnunet` |

---

## W&B Setup on DTU HPC

Log in once on a login node:
```bash
wandb login
```

This writes credentials to `~/.netrc`. They persist across jobs. Alternatively, set
`WANDB_API_KEY` in your job script.

To run offline (e.g. if the compute node has no internet):
```bash
export WANDB_MODE=offline
```

Then sync after the job completes:
```bash
wandb sync $nnUNet_results/wandb/offline-run-*/
```

---

## Caveats

- **`self.local_rank == 0` guard:** All W&B calls must be gated on rank 0. In DDP mode,
  all processes share the same trainer instance and will otherwise log duplicates.
- **`resume="allow"`:** If a job is killed and restarted with `--c`, W&B will append to
  the existing run rather than creating a new one. This requires the run ID to be stable
  — it is, because it is derived from the run name which includes the fold number.
- **Logger key names:** The internal `MetaLogger` dict keys (`"train_losses"`,
  `"val_losses"`, etc.) are hardcoded strings in `nnUNetTrainer`. If a future nnU-Net
  update renames them, the trainer will raise a `KeyError`. Pin the nnU-Net version in
  `pyproject.toml`.
- **BF16 and GradScaler:** When `autocast_dtype = torch.bfloat16`, the gradient scaler
  used internally by nnUNetTrainer is a no-op (BF16 does not underflow like FP16). This
  is harmless but can be explicitly disabled for clarity in a future refactor.
