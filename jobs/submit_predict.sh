#!/bin/bash
# Submit predict jobs (2d + 3d_fullres) for a dataset.
# Usage: bash jobs/submit_predict.sh DATASET_ID

set -euo pipefail

DATASET_ID=${1:?Usage: submit_predict.sh DATASET_ID}

mkdir -p jobs/logs

for CONFIG in 2d 3d_fullres; do
    sed "s/TPLCONFIG/${CONFIG}/g" jobs/predict.sh \
        | sed "s/DATASET_ID=1/DATASET_ID=${DATASET_ID}/" | bsub
    echo "Submitted: dataset=${DATASET_ID} config=${CONFIG}"
done
