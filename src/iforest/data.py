"""Data utilities: synthetic data for development and CSV loading."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_transactions(
    n_normal: int = 10_000,
    n_anomalies: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a labeled toy dataset that looks like card transactions.

    Features:
        amount      - transaction amount
        hour        - hour of day (0-23)
        merchant_risk - risk score of merchant (0-1)
        velocity    - transactions in last hour by this card
        distance_km - distance from home location

    Label: 1 = anomaly (fraud-like), 0 = normal.
    Labels exist ONLY for evaluation; the model never trains on them.
    """
    rng = np.random.default_rng(seed)

    normal = pd.DataFrame(
        {
            "amount": rng.lognormal(mean=3.2, sigma=0.6, size=n_normal),
            "hour": rng.normal(loc=13.0, scale=4.0, size=n_normal).clip(0, 23),
            "merchant_risk": rng.beta(a=2, b=8, size=n_normal),
            "velocity": rng.poisson(lam=1.2, size=n_normal),
            "distance_km": rng.exponential(scale=8.0, size=n_normal),
        }
    )
    normal["label"] = 0

    anomalies = pd.DataFrame(
        {
            # fraud pattern: large amounts, odd hours, risky merchants,
            # high velocity, far from home
            "amount": rng.lognormal(mean=6.5, sigma=0.5, size=n_anomalies),
            "hour": rng.choice([1, 2, 3, 4], size=n_anomalies).astype(float),
            "merchant_risk": rng.beta(a=8, b=2, size=n_anomalies),
            "velocity": rng.poisson(lam=8.0, size=n_anomalies),
            "distance_km": rng.exponential(scale=400.0, size=n_anomalies),
        }
    )
    anomalies["label"] = 1

    df = pd.concat([normal, anomalies], ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_csv(path: str, label_col: str | None = None) -> tuple[pd.DataFrame, pd.Series | None]:
    """Load a CSV, optionally splitting off a label column for evaluation."""
    df = pd.read_csv(path)
    labels = df.pop(label_col) if label_col and label_col in df.columns else None
    return df, labels
