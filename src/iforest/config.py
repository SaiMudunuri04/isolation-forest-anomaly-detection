"""Model configuration.

All hyperparameters live here, never hardcoded in training or inference code.
This is the single source of truth that CI, experiments and production share.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for the Isolation Forest pipeline."""

    # --- Isolation Forest hyperparameters ---
    n_estimators: int = 200
    max_samples: int = 256  # subsample size per tree; 256 is the paper's sweet spot
    contamination: float = 0.02  # expected share of anomalies in training data
    max_features: float = 1.0
    bootstrap: bool = False
    random_state: int = 42
    n_jobs: int = -1

    # --- Artifacts / versioning ---
    model_version: str = "1.0.0"
    artifact_dir: Path = Path("artifacts")

    # --- Retraining policy ---
    # Refit when the production score distribution drifts beyond this threshold.
    drift_alert_threshold: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 < self.contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        if self.n_estimators < 10:
            raise ValueError("n_estimators should be >= 10 for stable scores")
        if self.max_samples < 2:
            raise ValueError("max_samples must be >= 2")

    @property
    def model_path(self) -> Path:
        return self.artifact_dir / f"iforest_v{self.model_version}.joblib"

    @property
    def metadata(self) -> dict:
        return {
            "model_version": self.model_version,
            "fitted_at": datetime.now(timezone.utc).isoformat(),
            "params": {
                "n_estimators": self.n_estimators,
                "max_samples": self.max_samples,
                "contamination": self.contamination,
                "max_features": self.max_features,
                "bootstrap": self.bootstrap,
                "random_state": self.random_state,
            },
        }
