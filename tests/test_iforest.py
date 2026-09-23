"""pytest suite: the contract the model must keep."""

import numpy as np
import pandas as pd

from iforest.config import ModelConfig
from iforest.data import make_synthetic_transactions
from iforest.evaluation import evaluate_with_labels
from iforest.monitoring import drift_check
from iforest.pipeline import AnomalyDetectionPipeline


def _trained_pipe(tmp_path):
    df = make_synthetic_transactions(n_normal=2000, n_anomalies=50)
    labels = df.pop("label")
    config = ModelConfig(contamination=0.03, artifact_dir=tmp_path, model_version="test")
    pipe = AnomalyDetectionPipeline(config).fit(df)
    return pipe, df, labels


def test_scores_separate_anomalies(tmp_path):
    """A known-fraud row must score far above a typical normal row."""
    pipe, X, labels = _trained_pipe(tmp_path)
    scores = pipe.anomaly_score(X)
    assert scores[labels.to_numpy() == 1].mean() > scores[labels.to_numpy() == 0].mean()


def test_flag_rate_matches_contamination(tmp_path):
    """Flagged share on training data should be ~contamination."""
    pipe, X, _ = _trained_pipe(tmp_path)
    flag_rate = pipe.predict(X).mean()
    assert abs(flag_rate - 0.03) < 0.01, flag_rate


def test_serialization_roundtrip(tmp_path):
    """Save -> load must reproduce identical scores (train/serve parity)."""
    pipe, X, _ = _trained_pipe(tmp_path)
    path = pipe.save()
    reloaded = AnomalyDetectionPipeline.load(path)
    np.testing.assert_allclose(pipe.anomaly_score(X), reloaded.anomaly_score(X))


def test_evaluation_metrics_sane(tmp_path):
    pipe, X, labels = _trained_pipe(tmp_path)
    metrics = evaluate_with_labels(
        pipe.anomaly_score(X), labels.to_numpy(), pipe.threshold_, k=50
    )
    assert metrics["precision"] > 0.5, metrics
    assert metrics["recall"] > 0.5, metrics
    assert metrics["precision_at_50"] > 0.9, metrics


def test_drift_check_detects_shift(tmp_path):
    pipe, X, _ = _trained_pipe(tmp_path)
    train_scores = pipe.anomaly_score(X)
    shifted = X.copy()
    shifted["amount"] = shifted["amount"] * 10  # distribution shock
    prod_scores = pipe.anomaly_score(shifted)
    verdict = drift_check(train_scores, prod_scores)["verdict"]
    assert verdict in ("WATCH", "RETRAIN"), verdict


def test_missing_values_handled(tmp_path):
    pipe, X, _ = _trained_pipe(tmp_path)
    X_missing = X.copy()
    X_missing.iloc[:10, 0] = np.nan
    scores = pipe.anomaly_score(X_missing)
    assert np.isfinite(scores).all()


def test_unfitted_raises():
    pipe = AnomalyDetectionPipeline(ModelConfig())
    try:
        pipe.anomaly_score(pd.DataFrame({"a": [1.0]}))
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError for unfitted pipeline")
