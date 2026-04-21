#!/bin/bash
# Download MRI-CORE ViT-B pretrained weights to the location expected by
# src/probe/encoders/mri_core.py (MRI_CORE_WEIGHTS).
#
# Upstream: https://github.com/mazurowski-lab/mri_foundation (Apache-2.0)
# Paper:    arXiv:2506.12186
#
# Usage:
#     bash jobs/fetch_mri_core.sh
#
# Honours MODELS_DIR from the environment (same variable the Pydantic settings
# in src/utils/settings.py read). Default falls back to /work3/s225224/models
# to match that settings default. No download is performed if the target file
# already exists.

set -euo pipefail

DRIVE_FILE_ID="1nPkTI3H0vsujlzwY8jxjKwAbOCTJv4yW"
MIN_BYTES=$((100 * 1024 * 1024))  # 100 MB floor; Drive HTML-confirm page is ~4 KB

export UV_ENV_FILE=".env"

MODELS_DIR_RESOLVED="${MODELS_DIR:-/work3/s225224/models}"
TARGET_DIR="${MODELS_DIR_RESOLVED}/mri_core"
TARGET_FILE="${TARGET_DIR}/MRI_CORE_vitb.pth"

if [[ -f "${TARGET_FILE}" ]]; then
    existing_bytes=$(stat -c %s "${TARGET_FILE}")
    echo "Already present: ${TARGET_FILE} (${existing_bytes} bytes)"
    exit 0
fi

mkdir -p "${TARGET_DIR}"

echo "Downloading MRI-CORE weights to ${TARGET_FILE}"
uv run --with gdown gdown "${DRIVE_FILE_ID}" -O "${TARGET_FILE}"

actual_bytes=$(stat -c %s "${TARGET_FILE}")
if (( actual_bytes < MIN_BYTES )); then
    echo "Downloaded file is suspiciously small (${actual_bytes} bytes) — probably a Drive confirmation HTML page rather than the .pth. Removing."
    rm "${TARGET_FILE}"
    exit 1
fi

echo "Done (${actual_bytes} bytes). Saved to ${TARGET_FILE}"
