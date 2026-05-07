#!/bin/bash
# Find best configuration, ensemble, and postprocess predictions for a dataset.
# CPU only — run on login node after both predict jobs finish.
# Usage: bash jobs/ensemble_and_postprocess.sh DATASET_ID

set -euo pipefail

export UV_ENV_FILE=".env"
source .env

DATASET_ID=${1:?Usage: ensemble_and_postprocess.sh DATASET_ID}
DATASET_NAME=$(uv run python -c "from src.nnunet import DATASETS; print(DATASETS[${DATASET_ID}]['name'])")
PLANS=nnUNetResEncUNetLPlans
TRAINER=nnUNetTrainerWandB

RESULT_DIR="${nnUNet_results}/${DATASET_NAME}"

echo "=== Post-training pipeline: dataset=${DATASET_ID} (${DATASET_NAME}) ==="

# 1. Find best configuration
echo "--- find_best_configuration ---"
uv run nnUNetv2_find_best_configuration ${DATASET_ID} -c 2d 3d_fullres -p ${PLANS} -tr ${TRAINER}

# 2. Ensemble 2d + 3d_fullres predictions
echo "--- ensemble ---"
uv run nnUNetv2_ensemble \
    -i "${RESULT_DIR}/predictions_test_2d" "${RESULT_DIR}/predictions_test_3d_fullres" \
    -o "${RESULT_DIR}/predictions_test_ensemble" \
    -np 8

# 3. Apply postprocessing
echo "--- postprocessing ---"
ENSEMBLE_DIR="${RESULT_DIR}/ensembles/ensemble___${TRAINER}__${PLANS}__2d___${TRAINER}__${PLANS}__3d_fullres___0_1_2_3_4"
uv run nnUNetv2_apply_postprocessing \
    -i  "${RESULT_DIR}/predictions_test_ensemble" \
    -o  "${RESULT_DIR}/predictions_test_pp" \
    -pp_pkl_file "${ENSEMBLE_DIR}/postprocessing.pkl" \
    -np 8 \
    -plans_json "${ENSEMBLE_DIR}/plans.json"

echo "=== Done: ${RESULT_DIR}/predictions_test_pp/ ==="
