"""Evaluation: how you prove the model works before it touches production.

Unsupervised models still get rigorous evaluation:
  1. If ANY labels exist (fraud reports, incident tickets), use them -> precision/recall.
  2. Always report the score distribution and threshold sanity.
  3. Report precision@k: of the top-k riskiest records, how many were real issues.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

log = logging.getLogger(__name__)


def evaluate_with_labels(
    scores: np.ndarray, labels: np.ndarray, threshold: float, k: int = 100
) -> dict:
    """Evaluate anomaly scores against ground-truth labels (labels: 1=anomaly)."""
    labels = np.asarray(labels).astype(int)
    preds = (scores >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # precision@k: fraction of the k highest-scoring records that are true anomalies
    k = min(k, len(scores))
    top_k_idx = np.argsort(scores)[-k:]
    precision_at_k = float(labels[top_k_idx].mean())

    return {
        "threshold": float(threshold),
        "flag_rate": float(preds.mean()),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "average_precision": round(float(average_precision_score(labels, scores)), 4),
        f"precision_at_{k}": round(precision_at_k, 4),
        "n_flagged": int(preds.sum()),
        "n_true_anomalies": int(labels.sum()),
    }


def score_distribution_report(scores: np.ndarray) -> dict:
    """Sanity report on the score distribution; run on train AND production data."""
    return {
        "count": int(len(scores)),
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
        "min": round(float(scores.min()), 4),
        "p50": round(float(np.percentile(scores, 50)), 4),
        "p90": round(float(np.percentile(scores, 90)), 4),
        "p99": round(float(np.percentile(scores, 99)), 4),
        "max": round(float(scores.max()), 4),
    }


def threshold_tradeoff(scores: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Precision/recall at every candidate threshold, so the business can pick
    its operating point (fraud team staffing determines affordable alert volume)."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # precision_recall_curve returns one fewer threshold than precision/recall points
    return pd.DataFrame(
        {
            "threshold": np.append(thresholds, np.nan),
            "precision": precision,
            "recall": recall,
        }
    ).dropna()
