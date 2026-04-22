#!/bin/bash
# Re-submit 3d_fullres training jobs (all 5 folds).
# Usage: bash jobs/submit_3d.sh

set -euo pipefail

FOLDS="0 1 2 3 4"

mkdir -p jobs/logs

for fold in ${FOLDS}; do
    sed "s/TPLCONFIG/3d_fullres/g; s/TPLFOLD/${fold}/g" \
        jobs/train.sh | bsub
    echo "Submitted: config=3d_fullres fold=${fold}"
done
