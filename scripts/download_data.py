"""Download the UCI 'Default of Credit Card Clients' dataset (id 350) to data/raw/.

    python scripts/download_data.py

Source:  https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
License: CC BY 4.0 (credit: Yeh, I-Cheng, 2009, UCI Machine Learning Repository,
         https://doi.org/10.24432/C55S3H)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucimlrepo import fetch_ucirepo  # noqa: E402

from creditguard import config  # noqa: E402
from creditguard.data import standardize_columns  # noqa: E402


def main() -> None:
    dataset = fetch_ucirepo(id=350)
    raw = dataset.data.features.copy()
    targets = dataset.data.targets
    raw[targets.columns[0]] = targets.iloc[:, 0]
    print("Columns as delivered by ucimlrepo:", list(raw.columns))
    feature_cols = [c for c in raw.columns if c != targets.columns[0]]
    if len(feature_cols) == 23 and "X1" not in feature_cols and "LIMIT_BAL" not in feature_cols:
        # Fallback: names are neither X1..X23 nor the Kaggle style. The UCI column
        # ORDER is fixed (X1..X23), so rename by position. Check the printout above!
        print("WARNING: unexpected column names; renaming by position.")
        raw = raw.rename(columns={old: f"X{i + 1}" for i, old in enumerate(feature_cols)})
    df = standardize_columns(raw)
    config.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.DATA_PATH, index=False)
    print(f"Saved {len(df):,} rows x {df.shape[1]} columns -> {config.DATA_PATH}")
    print(f"Default rate: {df[config.TARGET].mean():.4f}")


if __name__ == "__main__":
    main()
