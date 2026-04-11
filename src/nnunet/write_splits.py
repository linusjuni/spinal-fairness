"""
Create a demographically stratified splits_final.json for nnU-Net 5-fold CV.

Reads split_v3.tsv and case_id_mapping.json, then creates 5 folds where each
fold preserves race_bin x age_bin x sex_bin proportions. The output is written
to $nnUNet_preprocessed/Dataset001_CSpineSeg/splits_final.json.

Must be run AFTER nnUNetv2_plan_and_preprocess (which creates the preprocessed
directory) and BEFORE nnUNetv2_train.

Usage:
    uv run -m src.nnunet.write_splits
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

from src.data.schemas import Col
from src.data.splits import load_splits
from src.nnunet import DATASET_NAME, SPLIT_VERSION
from src.utils.logger import get_logger
from src.utils.settings import settings

logger = get_logger(__name__)

N_FOLDS = 5


def create_nnunet_splits() -> list[dict[str, list[str]]]:
    """Create 5-fold stratified splits for nnU-Net.

    Only train+val cases from split_v3 go into imagesTr/ and thus into the
    nnU-Net folds. Test cases are in imagesTs/ and never appear here.

    Returns:
        List of 5 dicts, each with 'train' and 'val' keys mapping to lists
        of nnU-Net case identifier strings.
    """
    dataset_dir = settings.nnUNet_raw / DATASET_NAME

    # Load case_id_mapping.json (written by prepare_dataset)
    mapping_path = dataset_dir / "case_id_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(
            f"case_id_mapping.json not found at {mapping_path}. "
            "Run `uv run -m src.nnunet.prepare_dataset` first."
        )
    mapping = json.loads(mapping_path.read_text())

    # Load split_v3 for stratification columns
    splits = load_splits(SPLIT_VERSION)

    # Filter to train+val only (what's in imagesTr/)
    train_val_sids = {
        r["series_submitter_id"]
        for r in mapping
        if r["split"] in ("train", "val")
    }

    train_val = splits.filter(
        pl.col(Col.SERIES_SUBMITTER_ID).is_in(train_val_sids)
    )

    # Build a lookup from series_submitter_id -> case_id
    sid_to_cid = {r["series_submitter_id"]: r["case_id"] for r in mapping}

    # Build stratum labels for stratified k-fold
    case_ids = []
    strata = []
    for row in train_val.iter_rows(named=True):
        sid = row[Col.SERIES_SUBMITTER_ID]
        cid = sid_to_cid.get(sid)
        if cid is None:
            continue
        case_ids.append(cid)
        strata.append(row["stratum"])

    case_ids = np.array(case_ids)
    strata = np.array(strata)

    logger.info(
        "Stratifying",
        n_cases=len(case_ids),
        n_strata=len(np.unique(strata)),
        strata=sorted(np.unique(strata).tolist()),
    )

    # Stratified k-fold matching nnU-Net's default random_state
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=settings.RANDOM_SEED)

    folds = []
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(case_ids, strata)):
        train_ids = sorted(case_ids[train_indices].tolist())
        val_ids = sorted(case_ids[val_indices].tolist())
        folds.append({"train": train_ids, "val": val_ids})
        logger.info(
            f"Fold {fold_idx}",
            train=len(train_ids),
            val=len(val_ids),
        )

    return folds


def write_splits() -> None:
    """Create and write splits_final.json to the nnU-Net preprocessed directory."""
    preprocessed_dir = settings.nnUNet_preprocessed / DATASET_NAME

    if not preprocessed_dir.exists():
        raise FileNotFoundError(
            f"Preprocessed directory not found: {preprocessed_dir}. "
            "Run `nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity` first."
        )

    folds = create_nnunet_splits()

    splits_path = preprocessed_dir / "splits_final.json"
    splits_path.write_text(json.dumps(folds, indent=2))
    logger.success("Wrote splits_final.json", path=str(splits_path), n_folds=len(folds))


if __name__ == "__main__":
    write_splits()
