"""Pure rendering functions for the probe workstream.

Functions here take data / figures in and save PNGs out. No model calls,
no data loading, no heavy computation — everything that needs a GPU or a
dataset happens in `pipeline.py`, `extract.py`, `probe.py`, or `preview.py`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.eda.report import EDAReport


def pca_scatter(
    df: pl.DataFrame,
    attribute: str,
    report: EDAReport,
    *,
    title_prefix: str = "",
) -> None:
    """Scatter pc1 vs. pc2 coloured by `attribute`, saved via the report."""
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
        title = (
            f"{title_prefix}PCA by {attribute}"
            if title_prefix
            else f"PCA by {attribute}"
        )
        ax.set_title(title)
        ax.legend(loc="best", fontsize=9)


def _for_display(slice_2d: np.ndarray) -> np.ndarray:
    """Rotate 90 degrees CCW so the superior-inferior axis runs vertically."""
    return np.rot90(slice_2d, k=1)


def preprocessing_preview_grid(
    rows: list[dict],
    *,
    out_path: Path,
    column_titles: list[str],
    dpi: int = 120,
) -> None:
    """Render a rows x len(column_titles) grid of grayscale slices.

    Each `row` is a dict: ``{"series_id": str, "images": list[np.ndarray]}``.
    The ``images`` list must have the same length as ``column_titles``.
    """
    n_rows = len(rows)
    n_cols = len(column_titles)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False
    )
    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title)
    for row_idx, row in enumerate(rows):
        for col_idx, img in enumerate(row["images"]):
            ax = axes[row_idx, col_idx]
            ax.imshow(_for_display(img), cmap="gray")
            ax.axis("off")
        axes[row_idx, 0].text(
            0.02,
            0.98,
            row["series_id"],
            transform=axes[row_idx, 0].transAxes,
            color="yellow",
            fontsize=8,
            va="top",
            ha="left",
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 2},
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
