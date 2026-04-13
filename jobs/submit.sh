#!/bin/bash
# Submit all training jobs: 5 folds x N configs.
# Usage: bash jobs/submit.sh

set -euo pipefail

CONFIGS="2d 3d_fullres"
FOLDS="0 1 2 3 4"

mkdir -p jobs/logs

for config in ${CONFIGS}; do
    for fold in ${FOLDS}; do
        sed "s/CONFIG_PLACEHOLDER/${config}/g; s/FOLD_PLACEHOLDER/${fold}/g; s/cspine_CONFIG_foldFOLD/cspine_${config}_fold${fold}/g; s/CONFIG_fold_FOLD/${config}_fold_${fold}/g" \
            jobs/train.sh | bsub
        echo "Submitted: config=${config} fold=${fold}"
    done
done
