import numpy as np
import pandas as pd
import pytest

from creditguard import config
from creditguard.data import clean_data, standardize_columns
from creditguard.features import ENGINEERED_COLUMNS, add_features, build_preprocessor
from creditguard.synthetic import make_synthetic


def test_standardize_handles_uci_x_names():
    base = make_synthetic(50)
    names = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", *config.PAY_COLS, *config.BILL_COLS, *config.PAYAMT_COLS]
    raw = base[names].copy()
    raw.columns = [f"X{i + 1}" for i in range(23)]
    raw["Y"] = base[config.TARGET].values
    out = standardize_columns(raw)
    assert set(config.INPUT_COLUMNS + config.AUDIT_COLUMNS + [config.TARGET]) <= set(out.columns)
    pd.testing.assert_series_equal(out["PAY_1"], base["PAY_1"], check_names=False)


def test_standardize_handles_kaggle_names():
    base = make_synthetic(50).rename(columns={"PAY_1": "PAY_0", config.TARGET: "default.payment.next.month"})
    base.insert(0, "ID", range(len(base)))
    out = standardize_columns(base)
    assert "PAY_1" in out and config.TARGET in out and "ID" not in out


def test_standardize_missing_columns_raises():
    with pytest.raises(ValueError):
        standardize_columns(pd.DataFrame({"LIMIT_BAL": [1]}))


def test_clean_folds_undocumented_education_codes():
    df = make_synthetic(500)
    cleaned = clean_data(df)
    assert set(cleaned["EDUCATION"].unique()) <= {1, 2, 3, 4}


def test_add_features_creates_columns_without_nans_or_infs():
    df = clean_data(make_synthetic(300))
    df.loc[0, "LIMIT_BAL"] = 0  # division-by-zero guard
    out = add_features(df[config.INPUT_COLUMNS])
    assert all(c in out for c in ENGINEERED_COLUMNS)
    assert not np.isinf(out[ENGINEERED_COLUMNS].to_numpy()).any()
    assert not out[ENGINEERED_COLUMNS].isna().any().any()


def test_engineered_values_are_correct_on_a_known_row():
    row = pd.DataFrame([{
        "LIMIT_BAL": 100, "EDUCATION": 2,
        **{f"PAY_{i}": v for i, v in zip(range(1, 7), [2, 0, 0, 0, 0, -1])},
        **{f"BILL_AMT{i}": 50 for i in range(1, 7)},
        **{f"PAY_AMT{i}": 25 for i in range(1, 7)},
    }])
    out = add_features(row).iloc[0]
    assert out["util_last"] == pytest.approx(0.5)
    assert out["pay_ratio_last"] == pytest.approx(0.5)
    assert out["delay_max"] == 2
    assert out["n_late_months"] == 1
    assert out["delay_trend"] == 3  # 2 - (-1)


def test_preprocessor_never_uses_audit_columns():
    df = clean_data(make_synthetic(200))
    prep = build_preprocessor().fit(df.drop(columns=config.TARGET))
    names = " ".join(prep.named_steps["columns"].get_feature_names_out())
    for col in config.AUDIT_COLUMNS:
        assert col not in names


def test_negative_bill_flag():
    row = pd.DataFrame([{
        "LIMIT_BAL": 100, "EDUCATION": 2,
        **{f"PAY_{i}": 0 for i in range(1, 7)},
        **{f"BILL_AMT{i}": 50 for i in range(1, 7)},
        **{f"PAY_AMT{i}": 0 for i in range(1, 7)},
    }])
    assert add_features(row).iloc[0]["any_negative_bill"] == 0
    row.loc[0, "BILL_AMT3"] = -10
    assert add_features(row).iloc[0]["any_negative_bill"] == 1


def test_clean_data_keeps_duplicate_rows():
    df = make_synthetic(100)
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    assert len(clean_data(df)) == len(df)
