"""Feature engineering and the preprocessing pipeline.

``add_features`` is stateless (row-wise arithmetic only), so it cannot leak
information from the test set. Anything that LEARNS from data (scaling,
encoding) lives inside the sklearn Pipeline and is fit on training folds only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from creditguard import config

ENGINEERED_COLUMNS = [
    "util_last",        # last bill / credit limit
    "util_mean",        # average bill / credit limit
    "pay_ratio_last",   # last payment / last bill (bill floored at 1)
    "pay_ratio_mean",   # total paid / total billed (bill floored at 1)
    "delay_max",        # worst repayment status over 6 months
    "delay_mean",       # average repayment status
    "n_late_months",    # months with a delay (status >= 1)
    "delay_trend",      # recent minus oldest status (positive = getting worse)
    "bill_trend",       # last bill minus oldest bill
    "any_negative_bill",  # 1 if any statement was a credit balance (EDA: 16.5% vs 22.5% default)
]

NUMERIC_FEATURES = [
    "LIMIT_BAL", *config.PAY_COLS, *config.BILL_COLS, *config.PAYAMT_COLS, *ENGINEERED_COLUMNS
]
CATEGORICAL_FEATURES = ["EDUCATION"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with engineered columns appended."""
    out = df.copy()
    limit = out["LIMIT_BAL"].clip(lower=1)  # guard against division by zero
    bills = out[config.BILL_COLS]
    pays = out[config.PAYAMT_COLS]
    delays = out[config.PAY_COLS]

    out["util_last"] = out["BILL_AMT1"] / limit
    out["util_mean"] = bills.mean(axis=1) / limit
    out["pay_ratio_last"] = out["PAY_AMT1"] / out["BILL_AMT1"].clip(lower=1)
    out["pay_ratio_mean"] = pays.sum(axis=1) / bills.sum(axis=1).clip(lower=1)
    out["delay_max"] = delays.max(axis=1)
    out["delay_mean"] = delays.mean(axis=1)
    out["n_late_months"] = (delays >= 1).sum(axis=1)
    out["delay_trend"] = out["PAY_1"] - out["PAY_6"]
    out["bill_trend"] = out["BILL_AMT1"] - out["BILL_AMT6"]
    out["any_negative_bill"] = (bills < 0).any(axis=1).astype(int)

    # Ratios can explode for tiny bills; cap them so scaling stays sensible.
    for col in ("pay_ratio_last", "pay_ratio_mean"):
        out[col] = out[col].clip(upper=10)
    return out.replace([np.inf, -np.inf], np.nan)


RAW_NUMERIC_FEATURES = ["LIMIT_BAL", *config.PAY_COLS, *config.BILL_COLS, *config.PAYAMT_COLS]


def build_preprocessor(engineer: bool = True, pay_onehot: bool = False) -> Pipeline:
    """Feature engineering + imputation/scaling/encoding, as one unfitted pipeline.

    The defaults give the production pipeline. The two switches exist only for
    experiments (see creditguard.experiments):
      engineer=False   -> raw columns only, no engineered features
      pay_onehot=True  -> repayment statuses PAY_1..PAY_6 one-hot encoded instead of
                          numeric (rare statuses are grouped into one bucket)
    """
    numeric_cols = list(NUMERIC_FEATURES if engineer else RAW_NUMERIC_FEATURES)
    if pay_onehot:
        numeric_cols = [c for c in numeric_cols if c not in config.PAY_COLS]

    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),  # safety net; fit on train only
            ("scale", StandardScaler()),
        ]
    )
    transformers = [
        ("num", numeric, numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ]
    if pay_onehot:
        transformers.append(
            (
                "pay_cat",
                OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100),
                list(config.PAY_COLS),
            )
        )
    columns = ColumnTransformer(transformers, remainder="drop")  # drops SEX/AGE/MARRIAGE etc.
    first = FunctionTransformer(add_features, validate=False) if engineer else FunctionTransformer(validate=False)
    return Pipeline([("features", first), ("columns", columns)])
