#!/bin/bash
#BSUB -J cspine_val_TPLCONFIG_TPLFOLD
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 4
#BSUB -W 04:00
#BSUB -o jobs/logs/val_TPLCONFIG_TPLFOLD_%J.out
#BSUB -e jobs/logs/val_TPLCONFIG_TPLFOLD_%J.err

set -euo pipefail

DATASET_ID=1
CONFIG=TPLCONFIG
FOLD=TPLFOLD
PLANS=nnUNetResEncUNetLPlans
TRAINER=nnUNetTrainerWandB

export UV_ENV_FILE=".env"
export OMP_NUM_THREADS=1
export nnUNet_n_proc_DA=2
export nnUNet_wandb_enabled=0

mkdir -p jobs/logs

uv sync

TRAINER_DST="$(uv run python -c "import nnunetv2; print(nnunetv2.__path__[0])")/training/nnUNetTrainer/variants/training_length/nnUNetTrainerWandB.py"
cp "${TRAINER_SRC:-src/nnunet/trainer.py}" "${TRAINER_DST}"
echo "Installed custom trainer to ${TRAINER_DST}"

echo "=== Validation: config=${CONFIG} fold=${FOLD} ==="

uv run nnUNetv2_train \
    ${DATASET_ID} \
    ${CONFIG} \
    ${FOLD} \
    --npz \
    -p ${PLANS} \
    -tr ${TRAINER} \
    --val

echo "=== Validation complete: config=${CONFIG} fold=${FOLD} ==="
