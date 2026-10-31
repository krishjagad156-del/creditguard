import joblib
import numpy as np
import pandas as pd
import pytest

from creditguard import config
from creditguard.evaluation import choose_threshold, expected_cost, fairness_audit
from creditguard.synthetic import make_synthetic


def test_artifacts_and_metadata(trained):
    meta = trained["meta"]
    assert trained["model"].exists() and trained["meta_path"].exists()
    assert 0 < meta["threshold"] < 1
    assert meta["trained_on_synthetic_data"] is True
    assert meta["input_columns"] == config.INPUT_COLUMNS
    for key in ("roc_auc", "pr_auc", "brier", "recall", "precision"):
        assert key in meta["test_metrics"]


def test_model_beats_the_baseline_in_cv(trained):
    rows = {r["model"]: r for r in trained["meta"]["cv_comparison"]}
    assert rows["baseline_prior"]["cv_roc_auc_mean"] == pytest.approx(0.5)
    best = max(v["cv_roc_auc_mean"] for k, v in rows.items() if k != "baseline_prior")
    assert best > 0.6


def test_saved_pipeline_reloads_and_predicts(trained):
    model = joblib.load(trained["model"])
    df = make_synthetic(20, seed=7)
    probs = model.predict_proba(df[config.INPUT_COLUMNS].assign(EDUCATION=lambda d: d.EDUCATION.replace({0: 4, 5: 4, 6: 4})))[:, 1]
    assert probs.shape == (20,) and ((probs >= 0) & (probs <= 1)).all()


def test_threshold_prefers_recall_when_false_negatives_cost_more():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(y * 0.3 + rng.random(2000) * 0.7, 0, 1)
    cheap = choose_threshold(y, p, cost_fn=1, cost_fp=1)
    costly_fn = choose_threshold(y, p, cost_fn=10, cost_fp=1)
    assert costly_fn <= cheap


def test_expected_cost_simple_case():
    # 1 FN (cost 5) + 1 FP (cost 1) over 4 customers
    assert expected_cost([1, 0, 0, 0], [0, 1, 0, 0], cost_fn=5, cost_fp=1) == pytest.approx(1.5)


def test_fairness_audit_reports_each_group():
    y = [1, 0, 1, 0]
    p = [0.9, 0.1, 0.2, 0.8]
    g = pd.Series(["a", "a", "b", "b"])
    rows = {r["group"]: r for r in fairness_audit(y, p, g, threshold=0.5)}
    assert rows["a"]["recall"] == 1.0 and rows["a"]["false_positive_rate"] == 0.0
    assert rows["b"]["recall"] == 0.0 and rows["b"]["false_positive_rate"] == 1.0


def test_operating_points_in_metadata(trained):
    ops = trained["meta"]["operating_points"]
    defaults = [o for o in ops if o["is_default"]]
    assert len(defaults) == 1
    assert defaults[0]["threshold"] == pytest.approx(trained["meta"]["threshold"])
    rates = [o["test_flag_rate"] for o in ops]
    assert rates == sorted(rates)                      # sorted from strict to broad
    assert all(0.01 <= o["threshold"] <= 0.99 for o in ops)


def test_build_operating_points_capacity_thresholds_decrease():
    from creditguard.evaluation import build_operating_points

    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 4000)
    p = np.clip(y * 0.3 + rng.random(4000) * 0.7, 0, 1)
    ops = build_operating_points(y[:2000], p[:2000], y[2000:], p[2000:])
    cap = {o["id"]: o["threshold"] for o in ops if o["kind"] == "capacity"}
    assert cap["capacity_10"] > cap["capacity_20"] > cap["capacity_30"]
    cost = [o for o in ops if o["kind"] == "cost_ratio"]
    thresholds = [o["threshold"] for o in sorted(cost, key=lambda o: o["cost_ratio"])]
    assert thresholds == sorted(thresholds, reverse=True)   # higher miss-cost -> lower cut-off


def test_menu_has_no_near_duplicate_entries(trained):
    ids = {o["id"] for o in trained["meta"]["operating_points"]}
    assert "capacity_50" not in ids and "cost_1" not in ids
    assert {"cost_5", "capacity_10"} <= ids
    assert trained["meta"]["base_rate"] == pytest.approx(0.2, abs=0.1)
