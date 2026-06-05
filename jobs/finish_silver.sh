#!/bin/bash
# Finish the remaining silver (Dataset003) 3d_fullres folds so find_best_configuration
# can run. Per disk check (2026-06-05):
#   3d f0 -> has checkpoint_final but no validation/  -> validation-only (--val, minutes)
#   3d f1 -> already RUNNING (job 28602550)           -> NOT touched here
#   3d f4 -> no checkpoint at all                      -> fresh training (~44 h)
# Submit both now so f4 trains in parallel with the already-running f1 instead of after it.
#
# Usage: bash jobs/finish_silver.sh

set -euo pipefail

mkdir -p jobs/logs

# --- 3d_fullres fold 4: fresh training from scratch, fewer DA workers ---
# (original f4 run 28383789 died at epoch 0 with "background workers are no longer
#  alive"; n_proc_DA=1 hedges against that glibc/DA-worker bug.)
sed "s/TPLCONFIG/3d_fullres/g; s/TPLFOLD/4/g" jobs/train.sh \
    | sed 's/DATASET_ID=1/DATASET_ID=3/' \
    | sed 's/nnUNet_n_proc_DA=2/nnUNet_n_proc_DA=1/' \
    | bsub
echo "Submitted: dataset=3 (silver) config=3d_fullres fold=4 (fresh training, n_proc_DA=1)"

# --- 3d_fullres fold 0: validation-only from existing checkpoint_final (--val) ---
sed "s/TPLCONFIG/3d_fullres/g; s/TPLFOLD/0/g" jobs/validate.sh \
    | sed 's/DATASET_ID=1/DATASET_ID=3/' \
    | bsub
echo "Submitted: dataset=3 (silver) config=3d_fullres fold=0 validation-only (checkpoint_final)"
