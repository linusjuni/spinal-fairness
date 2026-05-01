#!/bin/bash
#BSUB -J cspine_predict_TPLCONFIG
#BSUB -q gpul40s
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -n 4
#BSUB -W 04:00
#BSUB -o jobs/logs/predict_TPLCONFIG_%J.out
#BSUB -e jobs/logs/predict_TPLCONFIG_%J.err

set -euo pipefail

# ---------- Config (replaced at submission) ----------
DATASET_ID=1
CONFIG=TPLCONFIG
PLANS=nnUNetResEncUNetLPlans
TRAINER=nnUNetTrainerWandB
# -----------------------------------------------------

export UV_ENV_FILE=".env"
export OMP_NUM_THREADS=1

mkdir -p jobs/logs

uv sync

# Install custom trainer so predict can load the checkpoint correctly.
TRAINER_SRC="src/nnunet/trainer.py"
TRAINER_DST="$(uv run python -c "import nnunetv2; print(nnunetv2.__path__[0])")/training/nnUNetTrainer/variants/training_length/nnUNetTrainerWandB.py"
cp "${TRAINER_SRC}" "${TRAINER_DST}"
echo "Installed custom trainer to ${TRAINER_DST}"

source .env

OUTPUT_DIR="${nnUNet_results}/Dataset00${DATASET_ID}_CSpineSeg/predictions_test_${CONFIG}"

echo "=== Predicting: dataset=${DATASET_ID} config=${CONFIG} ==="

uv run nnUNetv2_predict \
    -d Dataset00${DATASET_ID}_CSpineSeg \
    -i "${nnUNet_raw}/Dataset00${DATASET_ID}_CSpineSeg/imagesTs" \
    -o "${OUTPUT_DIR}" \
    -f 0 1 2 3 4 \
    -tr ${TRAINER} \
    -c ${CONFIG} \
    -p ${PLANS} \
    --save_probabilities

echo "=== Prediction complete: config=${CONFIG} output=${OUTPUT_DIR} ==="
