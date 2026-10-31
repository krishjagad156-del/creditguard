"""Loading, standardising and cleaning the UCI credit-card-default data.

The same dataset circulates with different column names (UCI download,
ucimlrepo, Kaggle mirror). ``standardize_columns`` maps all of them onto one
schema so the rest of the code never cares where the file came from.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from creditguard import config

# ucimlrepo / UCI description order: X1..X23 then Y.
_X_NAMES = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + config.PAY_COLS
    + config.BILL_COLS
    + config.PAYAMT_COLS
)
_X_RENAME = {f"X{i + 1}": name for i, name in enumerate(_X_NAMES)}

_OTHER_RENAME = {
    "PAY_0": "PAY_1",  # the original file skips PAY_1; we make the months sequential
    "default.payment.next.month": config.TARGET,
    "default payment next month": config.TARGET,
    "Y": config.TARGET,
}

REQUIRED_COLUMNS = [*config.INPUT_COLUMNS, *config.AUDIT_COLUMNS, config.TARGET]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the project schema and drop the ID column if present."""
    df = df.rename(columns={**_X_RENAME, **_OTHER_RENAME})
    df = df.drop(columns=[c for c in ("ID", "id") if c in df.columns])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df[REQUIRED_COLUMNS].copy()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply documented cleaning decisions (all confirmed against the real data in the EDA).

    EDUCATION: the UCI description lists codes 1-4, but the file also contains 0, 5
    and 6. Their default rates (0 -> 0.0%, 4 -> 5.7%, 5 -> 6.4%, 6 -> 15.7%, with
    n = 14, 123, 280, 51) are far below codes 1-3 (19-25%), so they are folded into
    4 ("other"). Code 6 rests on only 51 customers, so treat that as an assumption.

    Duplicates: 35 exact duplicate rows exist, ~86% of them all-zero accounts. There
    is no customer ID, so they may be different people with identical profiles. They
    are KEPT (7 of 6,000 test rows also appear in train, which is negligible).
    """
    df = df.copy()
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    return df


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    path = Path(path or config.DATA_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return standardize_columns(pd.read_csv(path))


def load_clean(path: Path | str | None = None) -> pd.DataFrame:
    return clean_data(load_raw(path))
