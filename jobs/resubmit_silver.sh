#!/bin/bash
# Resubmit the three silver (Dataset003) training runs that never finished.
# Usage: bash jobs/resubmit_silver.sh
#
#   config·fold | original job(s)        | failure                         | strategy
#   ------------|-----------------------|---------------------------------|------------------------
#   2d  · f1    | 28383782              | killed at epoch 925             | continue (--c)
#   3d  · f1    | 28383783, 28439543    | background-workers death @ ep 0 | from scratch, n_proc_DA=1
#   3d  · f4    | 28383789              | background-workers death @ ep 0 | from scratch, n_proc_DA=1
#
# `--c` continues from checkpoint_latest.pth (2d f1 has one ~epoch 900, so it
# redoes only ~100 epochs). The two 3d folds crashed during the first epoch with
# no usable checkpoint, so they start fresh. They hit the glibc/DA-worker bug that
# train.sh already mitigates (nnUNet_n_proc_DA=2); we drop to 1 here as an extra
# hedge since these specific folds failed twice / once on it.

set -euo pipefail

mkdir -p jobs/logs

# --- silver 2d fold 1: continue from checkpoint ---
sed "s/TPLCONFIG/2d/g; s/TPLFOLD/1/g" jobs/train.sh \
    | sed 's/DATASET_ID=1/DATASET_ID=3/' \
    | sed 's/-tr ${TRAINER}/-tr ${TRAINER} --c/' \
    | bsub
echo "Submitted: dataset=3 (silver) config=2d fold=1 (continue from checkpoint)"

# --- silver 3d_fullres folds 1 and 4: fresh start, fewer DA workers ---
for fold in 1 4; do
    sed "s/TPLCONFIG/3d_fullres/g; s/TPLFOLD/${fold}/g" jobs/train.sh \
        | sed 's/DATASET_ID=1/DATASET_ID=3/' \
        | sed 's/nnUNet_n_proc_DA=2/nnUNet_n_proc_DA=1/' \
        | bsub
    echo "Submitted: dataset=3 (silver) config=3d_fullres fold=${fold} (from scratch, n_proc_DA=1)"
done
