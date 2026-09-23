"""Train the model end to end.

Usage:
    python train.py                      # synthetic demo data
    python train.py --csv data.csv --label-col is_fraud
"""

from __future__ import annotations

import argparse
import json
import logging

from iforest.config import ModelConfig
from iforest.data import load_csv, make_synthetic_transactions
from iforest.evaluation import evaluate_with_labels, score_distribution_report
from iforest.explain import global_feature_importance
from iforest.pipeline import AnomalyDetectionPipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("train")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="CSV of training data (no labels needed)")
    ap.add_argument("--label-col", default=None, help="optional label column for evaluation")
    ap.add_argument("--contamination", type=float, default=0.02)
    ap.add_argument("--version", default="1.0.0")
    args = ap.parse_args()

    if args.csv:
        X, labels = load_csv(args.csv, label_col=args.label_col)
    else:
        log.info("No CSV given; generating synthetic transaction data")
        df = make_synthetic_transactions()
        labels = df.pop("label")
        X = df

    config = ModelConfig(
        contamination=args.contamination, model_version=args.version
    )
    pipe = AnomalyDetectionPipeline(config).fit(X)
    artifact = pipe.save()

    # --- evaluation (labels are optional) ---
    scores = pipe.anomaly_score(X)
    print("\n=== score distribution ===")
    print(json.dumps(score_distribution_report(scores), indent=2))

    if labels is not None:
        print("\n=== evaluation vs labels ===")
        print(
            json.dumps(
                evaluate_with_labels(
                    scores, labels.to_numpy(), pipe.threshold_, k=100
                ),
                indent=2,
            )
        )
        print("\n=== global feature importance ===")
        print(
            global_feature_importance(
                pipe.anomaly_score, X, labels=labels.to_numpy()
            ).to_string(index=False)
        )

    print(f"\nArtifact: {artifact}")
    print(f"Threshold: {pipe.threshold_:.4f}  (flag if score >= threshold)")


if __name__ == "__main__":
    main()
