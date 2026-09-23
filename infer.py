"""Score new data with a trained artifact.

Usage:
    python infer.py --model artifacts/iforest_v1.0.0.joblib --csv new_data.csv --out scored.csv
"""

from __future__ import annotations

import argparse
import logging

from iforest.data import load_csv
from iforest.pipeline import AnomalyDetectionPipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="scored.csv")
    ap.add_argument("--top", type=int, default=20, help="print top-N riskiest rows")
    args = ap.parse_args()

    pipe = AnomalyDetectionPipeline.load(args.model)
    X, _ = load_csv(args.csv)
    scored = pipe.predict_with_scores(X)
    scored.to_csv(args.out, index=False)

    print(f"Wrote {len(scored)} scored rows to {args.out}")
    print(f"Flagged {int(scored['is_anomaly'].sum())} anomalies "
          f"({scored['is_anomaly'].mean():.2%})")
    print(f"\nTop {args.top} riskiest:")
    print(scored.head(args.top).to_string(index=False))


if __name__ == "__main__":
    main()
