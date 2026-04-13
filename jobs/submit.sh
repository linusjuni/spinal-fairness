#!/bin/bash
# Submit all training jobs: 5 folds x N configs.
# Usage: bash jobs/submit.sh

set -euo pipefail

CONFIGS="2d 3d_fullres"
FOLDS="0 1 2 3 4"

mkdir -p jobs/logs

for config in ${CONFIGS}; do
    for fold in ${FOLDS}; do
        sed "s/TPLCONFIG/${config}/g; s/TPLFOLD/${fold}/g" \
            jobs/train.sh | bsub
        echo "Submitted: config=${config} fold=${fold}"
    done
done
