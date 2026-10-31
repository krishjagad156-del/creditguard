"""Synthetic data with the same schema as the real dataset.

ONLY for tests and smoke runs in environments without internet access.
Never report metrics from this data as real results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from creditguard import config


def make_synthetic(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "LIMIT_BAL": rng.choice([20_000, 50_000, 100_000, 200_000, 400_000], n),
            "SEX": rng.integers(1, 3, n),
            "EDUCATION": rng.choice([0, 1, 2, 3, 4, 5, 6], n, p=[0.01, 0.35, 0.45, 0.14, 0.03, 0.01, 0.01]),
            "MARRIAGE": rng.integers(0, 4, n),
            "AGE": rng.integers(21, 70, n),
        }
    )
    base_delay = rng.choice([-1, 0, 0, 0, 1, 2], n)
    for col in config.PAY_COLS:
        df[col] = np.clip(base_delay + rng.integers(-1, 2, n), -2, 8)
    for i in range(1, 7):
        df[f"BILL_AMT{i}"] = (df["LIMIT_BAL"] * rng.uniform(0, 0.9, n)).round()
        df[f"PAY_AMT{i}"] = (df[f"BILL_AMT{i}"] * rng.uniform(0, 0.5, n)).round()
    logit = -1.8 + 0.8 * df["PAY_1"] + 0.3 * df["PAY_2"] - 0.000002 * df["LIMIT_BAL"]
    df[config.TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df
