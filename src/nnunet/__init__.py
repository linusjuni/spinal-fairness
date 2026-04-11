"""
nnU-Net integration for the spinal-fairness project.

Modules:
    prepare_dataset — Build Dataset001_CSpineSeg under $nnUNet_raw.
    write_splits    — Create demographically stratified splits_final.json.
"""

DATASET_ID = 1
DATASET_NAME = "Dataset001_CSpineSeg"
SPLIT_VERSION = "split_v3"
