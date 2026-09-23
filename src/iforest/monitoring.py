"""Production monitoring: detect when the world changes under the model.

Anomaly detectors die silently: the input distribution drifts, the old
threshold keeps flagging 2% of traffic, but the 2% is no longer the right 2%.
This module compares live score distributions against the training baseline
and raises an alert when retraining is warranted.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import ks_2samp

log = logging.getLogger(__name__)


def population_stability_index(
    expected: np.ndarray, actual: np.ndarray, bins: int = 10
) -> float:
    """PSI between training scores (expected) and production scores (actual).

    Rule of thumb: < 0.1 no significant shift, 0.1-0.25 moderate, > 0.25 major.
    """
    breakpoints = np.histogram_bin_edges(expected, bins=bins)
    e_counts, _ = np.histogram(expected, bins=breakpoints)
    a_counts, _ = np.histogram(actual, bins=breakpoints)

    e_perc = e_counts / e_counts.sum()
    a_perc = a_counts / a_counts.sum()

    # avoid division by zero
    e_perc = np.clip(e_perc, 1e-6, None)
    a_perc = np.clip(a_perc, 1e-6, None)

    psi = float(np.sum((a_perc - e_perc) * np.log(a_perc / e_perc)))
    return psi


def drift_check(
    train_scores: np.ndarray,
    prod_scores: np.ndarray,
    alert_threshold: float = 0.1,
) -> dict:
    """Run PSI + Kolmogorov-Smirnov test; return a verdict dict."""
    psi = population_stability_index(train_scores, prod_scores)
    ks_stat, ks_p = ks_2samp(train_scores, prod_scores)

    verdict = "OK"
    if psi > 0.25:
        verdict = "RETRAIN"
    elif psi > alert_threshold:
        verdict = "WATCH"

    result = {
        "psi": round(psi, 4),
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue": float(f"{ks_p:.3g}"),
        "verdict": verdict,
    }
    log.info("Drift check: %s", result)
    return result
