#!/bin/bash
#BSUB -J fairness_evaluate
#BSUB -q hpc
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -n 24
#BSUB -W 02:00
#BSUB -o jobs/logs/fairness_evaluate_%J.out
#BSUB -e jobs/logs/fairness_evaluate_%J.err

set -euo pipefail

export UV_ENV_FILE=".env"

mkdir -p jobs/logs outputs

uv sync

source .env

echo "=== Evaluate: Dice + HD95, 24 workers ==="

uv run -m src.fairness.evaluate \
    --predictions "${nnUNet_results}/Dataset001_CSpineSeg/predictions_test_pp" \
    --references  "${nnUNet_raw}/Dataset001_CSpineSeg/labelsTs" \
    --mapping     "${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json" \
    --output      outputs/eval_global.csv \
    --metrics dice hd95 \
    --workers 24

echo "=== Analyze ==="

uv run -m src.fairness.analyze \
    --evaluation-csvs outputs/eval_global.csv \
    --ruler-labels    global \
    --mapping         "${nnUNet_raw}/Dataset001_CSpineSeg/case_id_mapping.json"

echo "=== Done ==="
