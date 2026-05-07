"""
nnU-Net integration for the spinal-fairness project.

Modules:
    prepare_dataset — Build nnU-Net dataset directories under $nnUNet_raw.
    write_splits    — Create demographically stratified splits_final.json.
"""

DATASETS = {
    1: {"name": "Dataset001_CSpineSeg", "split": "split_v3"},
    2: {"name": "Dataset002_CSpineSeg_Gold", "split": "split_v3_gold"},
    3: {"name": "Dataset003_CSpineSeg_Silver", "split": "split_v3_silver"},
}

DATASET_ID = 1
DATASET_NAME = DATASETS[1]["name"]
SPLIT_VERSION = DATASETS[1]["split"]
