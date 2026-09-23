"""Explainability: "why was this flagged?" is a production requirement.

Fraud analysts, auditors and customers all ask this. Two levels:
  1. GLOBAL - which features drive anomalies overall (permutation importance
     on the anomaly score).
  2. LOCAL  - for one flagged record, which features pushed it over the line
     (feature ablation: replace each feature with its median and measure how
     much the anomaly score drops).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def global_feature_importance(
    score_fn,
    X: pd.DataFrame,
    labels: np.ndarray | None = None,
    n_repeats: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Permutation importance for the anomaly model.

    WITH labels (preferred): drop in average precision when the feature is
    shuffled. Measures how much the feature contributes to *separating*
    anomalies from normal points.
    WITHOUT labels: mean absolute displacement of the anomaly scores.
    Measures how much the model *relies* on the feature.

    Note: with redundant features (several features each sufficient to spot
    the anomaly), single-feature permutation values are small by construction
    — the model does not need any one of them. That is information, not a bug.
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    has_labels = labels is not None

    if has_labels:
        baseline = float(average_precision_score(labels, score_fn(X)))

        def disruption(shuffled_scores: np.ndarray) -> float:
            return baseline - float(average_precision_score(labels, shuffled_scores))

    else:
        base_scores = score_fn(X)

        def disruption(shuffled_scores: np.ndarray) -> float:
            return float(np.mean(np.abs(shuffled_scores - base_scores)))

    importances: dict[str, float] = {}
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            X_shuffled = X.copy()
            X_shuffled[col] = rng.permutation(X_shuffled[col].to_numpy())
            drops.append(disruption(score_fn(X_shuffled)))
        importances[col] = max(float(np.mean(drops)), 0.0)

    out = pd.DataFrame(
        {"feature": list(importances), "importance": list(importances.values())}
    ).sort_values("importance", ascending=False)
    total = out["importance"].sum()
    if total > 0:
        out["importance"] = (out["importance"] / total).round(4)
    return out.reset_index(drop=True)


def explain_one_record(
    score_fn, X_train: pd.DataFrame, record: pd.Series
) -> pd.DataFrame:
    """Ablation-based local explanation for a single flagged record.

    For each feature, substitute the training median and see how far the
    anomaly score falls. Big drop => that feature's value drove the flag.
    Returns features ranked by contribution.
    """
    medians = X_train.median(numeric_only=True)
    base_score = float(score_fn(record.to_frame().T)[0])

    rows = []
    for col in record.index:
        ablated = record.copy()
        ablated[col] = medians[col]
        new_score = float(score_fn(ablated.to_frame().T)[0])
        rows.append(
            {
                "feature": col,
                "value": record[col],
                "median": medians[col],
                "score_drop": base_score - new_score,
            }
        )

    out = pd.DataFrame(rows).sort_values("score_drop", ascending=False)
    total = out["score_drop"].sum()
    out["contribution_pct"] = (out["score_drop"] / total * 100).round(1) if total else 0.0
    return out.reset_index(drop=True)
