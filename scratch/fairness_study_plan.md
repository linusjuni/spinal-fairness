# Fairness Study Plan

## Phase 1 — Data Preparation

1. **Build a metadata table** joining `case_RSNA*.tsv` (demographics) with `imaging_study_RSNA*.tsv` and `mr_series_RSNA*.tsv` (scanner info) into one row per exam.
2. **Create stratified train/val/test splits** (e.g. 70/10/20) ensuring balanced representation across demographic groups (age, sex, race, field strength).
3. **Prepare nnU-Net folder structure** — organize the NIfTI images and segmentation masks into the format nnU-Net expects.

## Phase 2 — Model Training

4. **Train nnU-Net** on the training split using 5-fold cross-validation (standard nnU-Net setup, 2D and 3D configurations).
5. **Track experiments** — log all hyperparameters, splits, and training curves.

## Phase 3 — Evaluation

6. **Run inference** on the held-out test set.
7. **Compute per-sample metrics** — Dice score and Hausdorff distance for each test exam, separately for vertebral bodies (label 1) and intervertebral discs (label 2).

## Phase 4 — Fairness Analysis

8. **Join metrics with demographics** — link each exam's scores to its patient's age, sex, race, ethnicity, scanner manufacturer, and field strength.
9. **Compare across subgroups** — test for statistically significant performance gaps (Mann-Whitney U, bootstrap CIs).
10. **Visualize** — boxplots, fairness gap tables, and per-group Dice distributions.

## Phase 5 — Gold vs Silver (blocked, waiting on authors)

11. **Separate Gold (expert) and Silver (auto) labels** once case IDs are received.
12. **Repeat the fairness analysis** using only Gold labels as ground truth.
13. **Compare** — does evaluating on Silver labels hide or exaggerate fairness gaps?
