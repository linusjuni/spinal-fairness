#!/bin/bash
# Resume the one gold (Dataset002) 2d fold-1 run that crashed at epoch 959
# (job 28383706, OSError: Stale file handle — a transient NFS error, not a
# code/data problem). nnUNet's `--c` continues from checkpoint_latest.pth
# (written at epoch 950 with save_every=50), so this redoes only ~50 epochs
# (~2-2.5 h at ~160 s/epoch) plus final validation, instead of ~44 h from scratch.
#
# Usage: bash jobs/resubmit_gold_2d_f1.sh
#
# Note: requires checkpoint_latest.pth to exist in
#   $nnUNet_results/Dataset002_CSpineSeg_Gold/.../2d/.../fold_1/
# If it is missing/corrupt, drop the `--c` injection below to train from scratch.

set -euo pipefail

mkdir -p jobs/logs

sed "s/TPLCONFIG/2d/g; s/TPLFOLD/1/g" jobs/train.sh \
    | sed 's/DATASET_ID=1/DATASET_ID=2/' \
    | sed 's/-tr ${TRAINER}/-tr ${TRAINER} --c/' \
    | bsub

echo "Submitted: dataset=2 (gold) config=2d fold=1 (continue from checkpoint)"
