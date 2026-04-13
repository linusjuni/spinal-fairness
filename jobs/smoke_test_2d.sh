#!/bin/bash
#BSUB -J cspine_smoke_2d
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 4
#BSUB -W 1:00
#BSUB -o jobs/cspine_smoke_2d_%J.out
#BSUB -e jobs/cspine_smoke_2d_%J.err

set -euo pipefail

# ---------- Config ----------
DATASET_ID=1
CONFIG=2d
FOLD=0
PLANS=nnUNetResEncUNetLPlans
# ----------------------------

export UV_ENV_FILE=".env"
export nnUNet_compile=false  # skip torch.compile for smoke test — re-enable for real training

uv sync

echo "Starting nnU-Net smoke test: config=${CONFIG}, fold=${FOLD}, plans=${PLANS}"

uv run nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} --npz -p ${PLANS}

echo "Smoke test completed."
