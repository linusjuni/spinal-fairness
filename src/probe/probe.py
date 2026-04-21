"""Linear probe for demographic signal in encoder embeddings.

Protocol (per docs/demographic-probing-of-medical-image-encoders/methodology.md,
section "Quantitative metrics"):

    PCA(n_pcs) inside the CV loop -> logistic regression with inner-CV C
    tuning -> outer-fold metric.

- Binary attribute (sex): AUROC across folds.
- Multi-class (age bin, race): balanced accuracy across folds.

The PCA is inside the pipeline so it refits per fold, avoiding any test-set
leakage. Returns mean +/- 95% CI across k folds.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as sps
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger
from src.utils.settings import settings

logger = get_logger(__name__)


def _build_pipeline(scoring: str, n_pcs: int, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("pca", PCA(n_components=n_pcs, random_state=seed)),
            (
                "clf",
                LogisticRegressionCV(
                    Cs=10,
                    cv=3,
                    penalty="l2",
                    max_iter=2000,
                    scoring=scoring,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def linear_probe(
    df: pl.DataFrame,
    emb_cols: list[str],
    attribute: str,
    *,
    n_pcs: int = 50,
    k_folds: int = 5,
    seed: int | None = None,
) -> dict:
    """Predict `attribute` from `emb_cols` via PCA -> logistic regression CV.

    Returns a dict with mean score, 95% CI, per-fold scores, and metadata.
    """
    if seed is None:
        seed = settings.RANDOM_SEED

    subset = df.filter(pl.col(attribute).is_not_null())
    X = subset.select(emb_cols).to_numpy()
    y = subset[attribute].to_numpy()

    classes, counts = np.unique(y, return_counts=True)
    n_classes = int(len(classes))
    if n_classes < 2:
        raise ValueError(f"Attribute {attribute!r} has only {n_classes} class")

    is_binary = n_classes == 2
    scoring = "roc_auc" if is_binary else "balanced_accuracy"
    metric_name = "auroc" if is_binary else "balanced_accuracy"
    effective_n_pcs = min(n_pcs, len(emb_cols), X.shape[0] - 1)

    pipeline = _build_pipeline(scoring=scoring, n_pcs=effective_n_pcs, seed=seed)
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    scores: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        pipeline.fit(X[train_idx], y[train_idx])
        if is_binary:
            proba = pipeline.predict_proba(X[test_idx])[:, 1]
            score = roc_auc_score(y[test_idx], proba)
        else:
            pred = pipeline.predict(X[test_idx])
            score = balanced_accuracy_score(y[test_idx], pred)
        scores.append(float(score))

    mean = float(np.mean(scores))
    var_hat = float(np.var(scores, ddof=1))

    # Nadeau-Bengio (2003, "Inference for the Generalization Error") corrected
    # variance for k-fold CV. The naive var_hat / k underestimates variance of
    # the CV mean because training folds overlap — NB inflates by (1/k + 1/(k-1))
    # over naive. For k=5 this makes the SE ~1.5x wider. CI is a Student-t with
    # k-1 dof. This is the right quantity when you want a CI around the CV mean
    # itself. See methodology.md; the field default (Glocker, Gichoya) is
    # instead a bootstrap on a held-out test set — inapplicable here since we
    # CV over the full cohort.
    t_crit = float(sps.t.ppf(0.975, df=k_folds - 1))
    se_naive = float(np.sqrt(var_hat / k_folds))
    se_nb = float(np.sqrt((1.0 / k_folds + 1.0 / (k_folds - 1)) * var_hat))
    ci95_naive = t_crit * se_naive
    ci95_nb = t_crit * se_nb

    logger.info(
        "Linear probe",
        attribute=attribute,
        metric=metric_name,
        mean=f"{mean:.3f}",
        ci95_nb=f"+/-{ci95_nb:.3f}",
        n=int(len(y)),
        n_classes=n_classes,
    )
    return {
        "attribute": attribute,
        "metric": metric_name,
        "n": int(len(y)),
        "n_classes": n_classes,
        "classes": [str(c) for c in classes],
        "class_counts": [int(c) for c in counts],
        "n_pcs": effective_n_pcs,
        "mean": mean,
        "ci95": ci95_nb,
        "ci95_naive": ci95_naive,
        "per_fold": scores,
    }
