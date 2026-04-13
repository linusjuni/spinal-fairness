"""
Custom nnU-Net trainer with enhanced W&B logging.

Layers on top of the built-in W&B integration (nnUNet_wandb_enabled=1).
Adds: progress.png uploads, checkpoint artifacts, epoch duration metric,
wandb.finish() on completion, and a hook point for future fairness metrics.

This file lives in our project tree. The job script copies it into the
nnunetv2 package tree at runtime so nnU-Net's trainer discovery finds it.

Usage:
    nnUNetv2_train 1 2d 0 --npz -tr nnUNetTrainerWandB
"""

from __future__ import annotations

from pathlib import Path

import torch
import wandb
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerWandB(nnUNetTrainer):

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def on_train_start(self) -> None:
        super().on_train_start()

        if wandb.run is not None:
            wandb.run.name = f"{self.configuration_name}_fold{self.fold}"

    def on_epoch_end(self) -> None:
        super().on_epoch_end()

        if wandb.run is None:
            return

        # current_epoch was already incremented by super()
        step = self.current_epoch - 1

        # Upload progress.png so we can see training curves in the W&B dashboard
        progress = Path(self.output_folder) / "progress.png"
        if progress.exists():
            wandb.log({"charts/progress": wandb.Image(str(progress))}, step=step)

        # Log epoch duration as a scalar (built-in only puts it in progress.png)
        t_start = self.logger.get_value("epoch_start_timestamps", step=step)
        t_end = self.logger.get_value("epoch_end_timestamps", step=step)
        wandb.log({"epoch_duration_s": t_end - t_start}, step=step)

    def on_train_end(self) -> None:
        super().on_train_end()

        if wandb.run is None:
            return

        # Log final checkpoint as a W&B artifact for reproducibility
        checkpoint = Path(self.output_folder) / "checkpoint_final.pth"
        if checkpoint.exists():
            artifact = wandb.Artifact(
                name=f"checkpoint-fold{self.fold}",
                type="model",
                metadata={
                    "configuration": self.configuration_name,
                    "fold": self.fold,
                    "plans": self.plans_manager.plans_name,
                },
            )
            artifact.add_file(str(checkpoint))
            wandb.log_artifact(artifact)

        wandb.finish()
