# Architecture — isolation-forest-anomaly-detection

## Design

The pipeline is built around a single invariant: **inference loads exactly
one artifact** (`iforest_vX.Y.Z.joblib`) that bundles the preprocessor, the
fitted Isolation Forest, the feature list, the calibrated decision
threshold, and training metadata. There is no separate "fit preprocessing"
step at serve time, so train/serve skew is impossible by construction.

Isolation Forest needs no feature scaling (splits are per-feature
thresholds), so preprocessing is only *select numeric columns → median
imputation*.

## Module map

- `config.py` — `ModelConfig`: immutable dataclass holding every
  hyperparameter (n_estimators=200, max_samples=256, contamination=0.02),
  artifact versioning, and the drift-alert threshold. Single source of truth
  shared by training, inference, and CI.
- `data.py` — `make_synthetic_transactions` (labeled toy data for
  development; labels are for evaluation only) and `load_csv` for real data.
- `pipeline.py` — `AnomalyDetectionPipeline`: `fit` / `anomaly_score` /
  `predict` / `predict_with_scores` / `save` / `load`. The threshold is
  calibrated from the contamination rate on training scores.
- `evaluation.py` — precision/recall/F1, precision@k, average precision,
  score-distribution report, threshold tradeoff curve.
- `monitoring.py` — PSI + Kolmogorov–Smirnov drift check comparing
  production score distributions to the training baseline; verdicts
  `OK` / `WATCH` / `RETRAIN`.
- `explain.py` — global permutation importance and per-record ablation, so
  every flagged anomaly ships with *why*.

## Deployment shape

This repo ships CLIs (`train.py`, `infer.py`), not an HTTP service, so the
Helm chart deploys a Kubernetes **CronJob**: scheduled retraining and batch
scoring. Runtime settings (contamination, model version, input CSV path)
come from a ConfigMap. For real-time scoring, load the artifact inside an
API with one `predict_with_scores` call per request.
