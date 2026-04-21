"""Probe visualisation: embeddings -> PCA(2) -> scatter by demographic.

Minimum viable first experiment (per
docs/demographic-probing-of-medical-image-encoders/sketch.md):
    encode cohort -> PCA -> plot coloured by sex / age bin / race.
If clusters are visible, the plot is the result.
"""

from __future__ import annotations

import polars as pl
from sklearn.decomposition import PCA

from src.data.groups import AgeStrategy, RaceStrategy, age, race
from src.data.loader import load_metadata
from src.data.schemas import Col
from src.eda.report import EDAReport
from src.probe.extract import load_embeddings
from src.utils.logger import get_logger
from src.utils.settings import settings

logger = get_logger(__name__)


def _embedding_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("emb_")]


def pca_2d(df: pl.DataFrame, emb_cols: list[str]) -> tuple[pl.DataFrame, list[float]]:
    """Fit PCA(n_components=2) on emb_cols; return (df with pc1, pc2, explained_ratio)."""
    X = df.select(emb_cols).to_numpy()
    model = PCA(n_components=2, random_state=settings.RANDOM_SEED)
    X2 = model.fit_transform(X)
    explained = [float(v) for v in model.explained_variance_ratio_]
    logger.info("PCA fit", n=X.shape[0], d=X.shape[1], explained=[round(v, 3) for v in explained])
    df = df.with_columns(
        pl.Series("pc1", X2[:, 0]),
        pl.Series("pc2", X2[:, 1]),
    )
    return df, explained


def scatter_by_attribute(
    df: pl.DataFrame,
    attribute: str,
    report: EDAReport,
    *,
    title_prefix: str = "",
) -> None:
    """Scatter pc1 vs. pc2 coloured by a categorical attribute."""
    pdf = df.select(["pc1", "pc2", attribute]).to_pandas()
    labels = pdf[attribute].astype(str).fillna("missing")
    with report.figure(f"pca_by_{attribute}", figsize=(7, 6)) as fig:
        ax = fig.gca()
        for label in sorted(labels.unique()):
            mask = labels == label
            ax.scatter(
                pdf.loc[mask, "pc1"],
                pdf.loc[mask, "pc2"],
                label=f"{label} (n={int(mask.sum())})",
                s=14,
                alpha=0.7,
            )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        title = f"{title_prefix}PCA by {attribute}" if title_prefix else f"PCA by {attribute}"
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9)


def run(encoder_name: str = "mri_core") -> None:
    """Load embeddings, join demographics, PCA(2), scatter by sex/age/race."""
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
        ),
        on=Col.SERIES_SUBMITTER_ID,
        how="inner",
    )

    emb_cols = _embedding_columns(df)
    df, explained = pca_2d(df, emb_cols)

    with EDAReport(encoder_name, report_type="probe") as report:
        report.log_stat("encoder", encoder_name)
        report.log_stat("n_samples", df.height)
        report.log_stat("output_dim", len(emb_cols))
        report.log_stat("pca_explained_variance_ratio", explained)
        for attr in [str(Col.SEX), f"{Col.RACE}_group", f"{Col.AGE}_group"]:
            scatter_by_attribute(df, attr, report, title_prefix=f"{encoder_name}: ")


if __name__ == "__main__":
    run("mri_core")
