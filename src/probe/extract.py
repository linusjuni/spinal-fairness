"""Encoder-agnostic batch feature extraction with Parquet caching.

Mirrors the staleness pattern in ``src.data.mri_volumes`` but also keys
off the encoder module's source mtime, so edits to an encoder's
preprocessing invalidate the cache automatically.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import torch

from src.data.exclusions import filter_excluded_cases
from src.data.loader import load_annotation_filenames
from src.data.schemas import Col
from src.probe.encoders import load_encoder
from src.utils.logger import get_logger
from src.utils.settings import settings

logger = get_logger(__name__)


def _cache_path(encoder_name: str) -> Path:
    return settings.processed_dir / f"embeddings_{encoder_name}.parquet"


def _encoder_module_path(encoder_name: str) -> Path:
    return Path(__file__).parent / "encoders" / f"{encoder_name}.py"


def _is_cache_fresh(cache_path: Path, encoder_name: str) -> bool:
    if not cache_path.exists():
        return False
    annotation_tsv = settings.structured_dir / "annotation_file_RSNA_20250321.tsv"
    encoder_module = _encoder_module_path(encoder_name)
    newest_source = max(
        annotation_tsv.stat().st_mtime,
        encoder_module.stat().st_mtime if encoder_module.exists() else 0.0,
    )
    return cache_path.stat().st_mtime > newest_source


def extract_embeddings(
    encoder_name: str,
    force_refresh: bool = False,
    device: str = "cuda",
) -> pl.DataFrame:
    """Encode every annotated exam and cache results to Parquet.

    Returns a DataFrame with columns:
        series_submitter_id, filename, emb_0, emb_1, ..., emb_{d-1}
    """
    cache_path = _cache_path(encoder_name)

    if not force_refresh and _is_cache_fresh(cache_path, encoder_name):
        df = pl.read_parquet(cache_path)
        logger.info("Loaded embeddings from cache", rows=df.height, encoder=encoder_name)
        return df

    enc = load_encoder(encoder_name, device=device)
    filenames = load_annotation_filenames()
    logger.info("Extracting embeddings", n=filenames.height, encoder=encoder_name)

    rows: list[dict] = []
    failed: list[tuple[str, str]] = []

    with torch.inference_mode():
        for i, row in enumerate(filenames.iter_rows(named=True)):
            filename = row[Col.FILENAME]
            series_id = row[Col.SERIES_SUBMITTER_ID]
            path = settings.annotation_dir / filename

            if i % 50 == 0:
                logger.info(f"Encoding {i}/{filenames.height}")

            if not path.exists():
                failed.append((filename, "file not found"))
                continue

            try:
                x = enc.preprocess(path).to(device).unsqueeze(0)
                feat = enc.model(x).squeeze(0).float().cpu().numpy()
            except Exception as e:
                logger.warning("Encoding failed", filename=filename, error=str(e))
                failed.append((filename, str(e)))
                continue

            rows.append(
                {
                    Col.SERIES_SUBMITTER_ID: series_id,
                    Col.FILENAME: filename,
                    **{f"emb_{j}": float(v) for j, v in enumerate(feat)},
                }
            )

    if not rows:
        raise RuntimeError(f"All {filenames.height} encodings failed")

    df = pl.DataFrame(rows)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache_path)
    logger.success(
        "Cached embeddings",
        rows=df.height,
        failed=len(failed),
        path=cache_path.name,
    )
    return df


def load_embeddings(
    encoder_name: str,
    force_refresh: bool = False,
    device: str = "cuda",
) -> pl.DataFrame:
    """Extract (or load from cache) and apply project-wide case exclusions."""
    df = extract_embeddings(encoder_name, force_refresh=force_refresh, device=device)
    return filter_excluded_cases(df, logger)
