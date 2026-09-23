"""Core training and inference pipeline.

The artifact saved to disk bundles EVERYTHING inference needs:
preprocessor + model + feature names + threshold + metadata.
There is no "fit preprocessing separately and hope they match" in production.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .config import ModelConfig

log = logging.getLogger(__name__)

# Isolation Forest needs no scaling (splits are per-feature thresholds),
# so preprocessing is only: select numerics -> median imputation.
# This keeps train/serve skew impossible by construction.


def build_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), feature_names),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class AnomalyDetectionPipeline:
    """Fit-once, score-anywhere wrapper around sklearn's IsolationForest."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.pipeline: Pipeline | None = None
        self.feature_names: list[str] = []
        self.threshold_: float | None = None  # calibrated decision threshold
        self.fitted_at: str | None = None

    # ---------------- training ----------------
    def fit(self, X: pd.DataFrame) -> "AnomalyDetectionPipeline":
        self.feature_names = list(X.columns)
        self.pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(self.feature_names)),
                (
                    "model",
                    IsolationForest(
                        n_estimators=self.config.n_estimators,
                        max_samples=self.config.max_samples,
                        contamination=self.config.contamination,
                        max_features=self.config.max_features,
                        bootstrap=self.config.bootstrap,
                        random_state=self.config.random_state,
                        n_jobs=self.config.n_jobs,
                    ),
                ),
            ]
        )
        log.info("Fitting IsolationForest on %d rows x %d features", *X.shape)
        self.pipeline.fit(X)

        # Calibrate the threshold on the TRAINING scores so inference uses a
        # fixed, auditable cutoff instead of recomputing quantiles per batch.
        train_scores = self._raw_scores(X)
        self.threshold_ = float(
            np.quantile(train_scores, 1.0 - self.config.contamination)
        )
        self.fitted_at = self.config.metadata["fitted_at"]
        log.info(
            "Calibrated threshold=%.4f (top %.1f%% of train scores flagged)",
            self.threshold_,
            self.config.contamination * 100,
        )
        return self

    # ---------------- inference ----------------
    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """Higher = more anomalous. Range is roughly [0, 1]."""
        self._check_fitted()
        return self._raw_scores(X)

    def _raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        # sklearn's score_samples is the OPPOSITE of the anomaly score
        # (lower = more abnormal), so we negate it.
        raw = self.pipeline.named_steps["model"].score_samples(
            self.pipeline.named_steps["preprocess"].transform(X)
        )
        return -raw

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """1 = anomaly, 0 = normal, using the calibrated training threshold."""
        return (self.anomaly_score(X) >= self.threshold_).astype(int)

    def predict_with_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["anomaly_score"] = self.anomaly_score(X)
        out["is_anomaly"] = self.predict(X)
        return out.sort_values("anomaly_score", ascending=False)

    # ---------------- persistence ----------------
    def save(self, path: Path | None = None) -> Path:
        self._check_fitted()
        path = Path(path or self.config.model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "feature_names": self.feature_names,
                "threshold": self.threshold_,
                "config": self.config,
                "fitted_at": self.fitted_at,
            },
            path,
        )
        log.info("Saved artifact to %s", path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetectionPipeline":
        blob = joblib.load(path)
        obj = cls(config=blob["config"])
        obj.pipeline = blob["pipeline"]
        obj.feature_names = blob["feature_names"]
        obj.threshold_ = blob["threshold"]
        obj.fitted_at = blob["fitted_at"]
        log.info("Loaded artifact %s (fitted at %s)", path, obj.fitted_at)
        return obj

    def _check_fitted(self) -> None:
        if self.pipeline is None or self.threshold_ is None:
            raise RuntimeError("Pipeline is not fitted yet. Call .fit() or .load() first.")
