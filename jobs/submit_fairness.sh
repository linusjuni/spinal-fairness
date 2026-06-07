#!/bin/bash
# Submit each fairness analysis as its own hpc job so they run in parallel.
# Usage: bash jobs/submit_fairness.sh [stage ...]
#   default stages: global biased_ruler bias_amplification
#   e.g. only the first two:  bash jobs/submit_fairness.sh global biased_ruler

set -euo pipefail

mkdir -p jobs/logs

STAGES="${*:-global biased_ruler bias_amplification}"

for stage in ${STAGES}; do
    sed "s/TPLSTAGE/${stage}/g" jobs/fairness_analysis.sh | bsub
    echo "Submitted: ${stage}"
done
