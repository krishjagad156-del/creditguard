import pytest

from creditguard import config
from creditguard.data import clean_data, standardize_columns
from creditguard.experiments import REFERENCE, run
from creditguard.features import ENGINEERED_COLUMNS, build_preprocessor
from creditguard.synthetic import make_synthetic


@pytest.fixture(scope="module")
def df():
    return clean_data(standardize_columns(make_synthetic(1500, seed=5)))


def _names(prep, df):
    prep.fit(df[config.INPUT_COLUMNS])
    return list(prep.named_steps["columns"].get_feature_names_out())


def test_default_preprocessor_includes_engineered_features(df):
    names = " ".join(_names(build_preprocessor(), df))
    assert all(c in names for c in ENGINEERED_COLUMNS)


def test_raw_only_has_no_engineered_features(df):
    names = " ".join(_names(build_preprocessor(engineer=False), df))
    assert not any(c in names for c in ENGINEERED_COLUMNS)
    assert "PAY_1" in names


def test_pay_onehot_replaces_numeric_pay_columns(df):
    names = _names(build_preprocessor(pay_onehot=True), df)
    assert "num__PAY_1" not in names
    assert any(n.startswith("pay_cat__PAY_1_") for n in names)


def test_no_variant_uses_audit_columns(df):
    for kwargs in ({}, {"engineer": False}, {"pay_onehot": True}):
        names = " ".join(_names(build_preprocessor(**kwargs), df))
        assert not any(c in names for c in config.AUDIT_COLUMNS)


def test_experiment_run_reports_reference_as_zero_diff(df, tmp_path):
    res = run(df=df, cv_folds=3, out_dir=tmp_path)
    assert len(res) == 6                                    # 2 models x 3 variants
    ref = res[res["variant"] == REFERENCE]
    assert (ref["pr_auc_diff_vs_current"].abs() < 1e-12).all()
    assert (tmp_path / "experiments.csv").exists()
