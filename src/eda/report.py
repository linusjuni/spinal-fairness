from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.figure import Figure

from src.utils.logger import get_logger
from src.utils.settings import settings


class EDAReport:
    """Output sink for EDA analyses. Saves figures, tables, and stats."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._root = settings.OUTPUT_DIR / "eda"
        self._stats: dict[str, Any] = {}
        self._logger = get_logger(f"eda.{name}")
        self._run_dir: Path | None = None

    def __enter__(self) -> EDAReport:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_dir = self._root / self._name / ts
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._logger.info("Report started", path=str(self._run_dir))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        assert self._run_dir is not None
        stats_path = self._run_dir / "stats.json"
        stats_path.write_text(json.dumps(self._stats, indent=2, default=str))
        self._logger.success(
            "Report finished",
            stats=len(self._stats),
            path=str(self._run_dir),
        )

    def save_fig(self, fig: Figure, name: str, **kwargs: Any) -> Path:
        """Save a matplotlib figure. kwargs override savefig defaults."""
        assert self._run_dir is not None
        defaults = {"dpi": 150, "bbox_inches": "tight"}
        defaults.update(kwargs)
        path = self._run_dir / f"{name}.png"
        fig.savefig(path, **defaults)
        plt.close(fig)
        self._logger.info("Saved figure", name=name)
        return path

    def save_table(self, df: pl.DataFrame, name: str) -> Path:
        """Save a polars DataFrame as CSV."""
        assert self._run_dir is not None
        path = self._run_dir / f"{name}.csv"
        df.write_csv(path)
        self._logger.info("Saved table", name=name, shape=str(df.shape))
        return path

    def log_stat(self, key: str, value: Any) -> None:
        """Accumulate a stat for the JSON dump at report close."""
        self._stats[key] = value

    @contextmanager
    def figure(
        self,
        name: str,
        *,
        figsize: tuple[float, float] | None = None,
        **save_kwargs: Any,
    ):
        """Context manager that creates a figure and auto-saves it on exit."""
        fig = plt.figure(figsize=figsize)
        try:
            yield fig
        finally:
            self.save_fig(fig, name, **save_kwargs)
