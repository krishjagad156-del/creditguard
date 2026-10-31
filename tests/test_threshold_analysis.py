import numpy as np

from creditguard import config
from creditguard.data import clean_data, standardize_columns
from creditguard.synthetic import make_synthetic
from creditguard.threshold_analysis import capacity_table, cost_ratio_table, run


def test_capacity_table_is_monotonic_and_consistent():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 3000)
    p = np.clip(y * 0.3 + rng.random(3000) * 0.7, 0, 1)
    t = capacity_table(y, p)
    assert t["recall"].is_monotonic_increasing          # reviewing more never catches fewer
    assert (t["defaulters_caught"] <= y.sum()).all()
    assert t["lift_vs_random"].iloc[0] > 1               # best-ranked customers beat random


def test_higher_fn_cost_flags_more_customers():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000)
    p = np.clip(y * 0.3 + rng.random(4000) * 0.7, 0, 1)
    t = cost_ratio_table(y[:2000], p[:2000], y[2000:], p[2000:], ratios=[1, 5, 20])
    assert t["flag_rate"].is_monotonic_increasing


def test_run_end_to_end_on_synthetic(trained, tmp_path):
    df = clean_data(standardize_columns(make_synthetic(2000)))  # same seed/size as the fixture
    res = run(model_path=trained["model"], df=df, out_dir=tmp_path, cv_folds=3)
    assert (tmp_path / "capacity_table.csv").exists()
    assert len(res["cost_ratio"]) == 6
