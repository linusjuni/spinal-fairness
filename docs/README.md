# Documentation Index

## nnU-Net Training Pipeline

| Document | Contents |
|---|---|
| [01 — Setup](nnunet/01_setup.md) | Installation (PyTorch-first), environment variables, directory layout, dependency gotchas |
| [02 — Dataset Preparation](nnunet/02_dataset_preparation.md) | nnU-Net format, file naming, dataset.json normalization, custom demographic splits |
| [03 — Training](nnunet/03_training.md) | plan_and_preprocess, anisotropy handling, ResEnc planners, training commands, BF16, DTU HPC job scripts |
| [04 — Experiment Tracking](nnunet/04_experiment_tracking.md) | Built-in W&B integration, custom trainer subclassing, MetaLogger API |
| [05 — Inference & Evaluation](nnunet/05_inference.md) | Model selection, prediction, postprocessing, evaluation (Dice/IoU), HD95 workaround, fairness analysis |
