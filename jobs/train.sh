#!/bin/bash
#BSUB -J cspine_TPLCONFIG_TPLFOLD
#BSUB -q gpua100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 4
#BSUB -W 72:00
#BSUB -o jobs/logs/TPLCONFIG_TPLFOLD_%J.out
#BSUB -e jobs/logs/TPLCONFIG_TPLFOLD_%J.err

set -euo pipefail

# ---------- Config (replaced by submit.sh) ----------
DATASET_ID=1
CONFIG=TPLCONFIG
FOLD=TPLFOLD
PLANS=nnUNetResEncUNetLPlans
TRAINER=nnUNetTrainerWandB
# -----------------------------------------------------

export UV_ENV_FILE=".env"

mkdir -p jobs/logs

uv sync

# Install the custom trainer into nnunetv2's package tree so trainer discovery finds it.
TRAINER_SRC="src/nnunet/trainer.py"
TRAINER_DST="$(uv run python -c "import nnunetv2; print(nnunetv2.__path__[0])")/training/nnUNetTrainer/variants/training_length/nnUNetTrainerWandB.py"
cp "${TRAINER_SRC}" "${TRAINER_DST}"
echo "Installed custom trainer to ${TRAINER_DST}"

echo "=== Training: dataset=${DATASET_ID} config=${CONFIG} fold=${FOLD} plans=${PLANS} trainer=${TRAINER} ==="

uv run nnUNetv2_train \
    ${DATASET_ID} \
    ${CONFIG} \
    ${FOLD} \
    --npz \
    -p ${PLANS} \
    -tr ${TRAINER}

echo "=== Training complete: config=${CONFIG} fold=${FOLD} ==="
