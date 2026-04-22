"""Probe pipeline orchestrator.

Loads cached embeddings for the named encoder, joins demographics, fits
PCA(2) for qualitative scatter plots, runs the linear probe (PCA(50) +
logistic regression CV) for each attribute, and writes everything as an
EDAReport under ``outputs/probe/{encoder_name}/{timestamp}/``.

Usage:
    uv run -m src.probe.pipeline <encoder_name>

Compute (PCA, probe) lives here; rendering lives in `plots.py`.
"""

from __future__ import annotations

import polars as pl
from sklearn.decomposition import PCA

from src.data.groups import AgeStrategy, RaceStrategy, age, race
from src.data.loader import load_metadata
from src.data.schemas import Col
from src.eda.report import EDAReport
from src.probe.extract import load_embeddings
from src.probe.plots import pca_scatter
from src.probe.probe import linear_probe
from src.utils.logger import get_logger
from src.utils.settings import settings

logger = get_logger(__name__)


def _embedding_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("emb_")]


def pca_2d(
    df: pl.DataFrame, emb_cols: list[str]
) -> tuple[pl.DataFrame, list[float]]:
    """Fit PCA(n_components=2) on emb_cols; return (df with pc1, pc2, explained_ratio)."""
    X = df.select(emb_cols).to_numpy()
    model = PCA(n_components=2, random_state=settings.RANDOM_SEED)
    X2 = model.fit_transform(X)
    explained = [float(v) for v in model.explained_variance_ratio_]
    logger.info(
        "PCA fit",
        n=X.shape[0],
        d=X.shape[1],
        explained=[round(v, 3) for v in explained],
    )
    return (
        df.with_columns(pl.Series("pc1", X2[:, 0]), pl.Series("pc2", X2[:, 1])),
        explained,
    )


def run(encoder_name: str) -> None:
    """Load embeddings, join demographics, PCA(2) scatters, linear probes.

    Scanner/manufacturer is included as a sanity attribute: a healthy
    pipeline should reach high AUROC on scanner since vendors produce
    distinct intensity fingerprints. If even scanner looks null, the
    upstream pipeline (slice axis, preprocessing) is suspect.
    """
    embeddings = load_embeddings(encoder_name)
    metadata = load_metadata()

    metadata = race[RaceStrategy.WHITE_VS_BLACK_VS_OTHER].apply(metadata, Col.RACE)
    metadata = age[AgeStrategy.THREE_BINS].apply(metadata, Col.AGE)

    df = embeddings.join(
        metadata.select(
            Col.SERIES_SUBMITTER_ID,
            Col.SEX,
            f"{Col.RACE}_group",
            f"{Col.AGE}_group",
            Col.MANUFACTURER,
        ),
        on=Col.SERIES_SUBMITTER_ID,
        how="inner",
    )

    emb_cols = _embedding_columns(df)
    df, explained = pca_2d(df, emb_cols)

    attributes = [
        str(Col.SEX),
        f"{Col.RACE}_group",
        f"{Col.AGE}_group",
        str(Col.MANUFACTURER),
    ]

    with EDAReport(encoder_name, report_type="probe") as report:
        report.log_stat("encoder", encoder_name)
        report.log_stat("n_samples", df.height)
        report.log_stat("output_dim", len(emb_cols))
        report.log_stat("pca_explained_variance_ratio", explained)

        for attr in attributes:
            pca_scatter(df, attr, report, title_prefix=f"{encoder_name}: ")

        probe_results = [linear_probe(df, emb_cols, attr) for attr in attributes]
        for res in probe_results:
            report.log_stat(f"probe_{res['attribute']}", res)

        probe_table = pl.DataFrame(
            [
                {
                    "attribute": r["attribute"],
                    "metric": r["metric"],
                    "n": r["n"],
                    "n_classes": r["n_classes"],
                    "n_pcs": r["n_pcs"],
                    "mean": round(r["mean"], 4),
                    "ci95": round(r["ci95"], 4),
                    "ci95_naive": round(r["ci95_naive"], 4),
                }
                for r in probe_results
            ]
        )
        report.save_table(probe_table, "probe_linear")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        sys.exit("usage: python -m src.probe.pipeline <encoder_name>")
    run(sys.argv[1])
