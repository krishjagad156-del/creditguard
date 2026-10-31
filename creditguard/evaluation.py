"""Metrics, cost-aware threshold selection and the fairness audit."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from creditguard import config


def expected_cost(y_true, y_pred, cost_fn: float, cost_fp: float) -> float:
    """Average cost per customer for a set of hard predictions."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float((fn * cost_fn + fp * cost_fp) / max(len(y_true), 1))


def choose_threshold(
    y_true,
    y_prob,
    cost_fn: float = config.COST_FALSE_NEGATIVE,
    cost_fp: float = config.COST_FALSE_POSITIVE,
) -> float:
    """Pick the probability cut-off that minimises expected cost.

    Call this with OUT-OF-FOLD training predictions, never with test data.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    grid = np.linspace(0.02, 0.98, 97)
    costs = [expected_cost(y_true, (y_prob >= t).astype(int), cost_fn, cost_fp) for t in grid]
    return float(grid[int(np.argmin(costs))])


def classification_report_at(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "flag_rate": float(y_pred.mean()),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "expected_cost_per_customer": expected_cost(
            y_true, y_pred, config.COST_FALSE_NEGATIVE, config.COST_FALSE_POSITIVE
        ),
    }


def threshold_free_metrics(y_true, y_prob) -> dict:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "base_rate": float(np.mean(y_true)),
    }


def calibration_table(y_true, y_prob, n_bins: int = 10) -> list[dict]:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return [
        {"mean_predicted": float(p), "fraction_positive": float(f)}
        for p, f in zip(mean_pred, frac_pos)
    ]


def age_band(age: pd.Series) -> pd.Series:
    return pd.cut(
        age, bins=[0, 29, 39, 49, 200], labels=["<30", "30-39", "40-49", "50+"]
    ).astype(str)


def fairness_audit(y_true, y_prob, groups: pd.Series, threshold: float) -> list[dict]:
    """Per-group error rates at the chosen threshold.

    This DESCRIBES gaps; it does not decide whether they are acceptable. That is
    a policy question (and a legal one in real lending). See the README.
    """
    frame = pd.DataFrame(
        {"y": np.asarray(y_true), "p": np.asarray(y_prob), "g": np.asarray(groups)}
    )
    rows = []
    for name, part in frame.groupby("g"):
        pred = (part["p"] >= threshold).astype(int)
        pos, neg = (part["y"] == 1).sum(), (part["y"] == 0).sum()
        tp = ((pred == 1) & (part["y"] == 1)).sum()
        fp = ((pred == 1) & (part["y"] == 0)).sum()
        rows.append(
            {
                "group": str(name),
                "n": int(len(part)),
                "base_rate": float(part["y"].mean()),
                "flag_rate": float(pred.mean()),
                "recall": float(tp / pos) if pos else None,
                "false_positive_rate": float(fp / neg) if neg else None,
                "precision": float(tp / (tp + fp)) if (tp + fp) else None,
            }
        )
    return rows


def audit_groups(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Group labels for the audited attributes (codes follow the UCI description)."""
    return {
        "SEX": df["SEX"].map({1: "male", 2: "female"}).fillna("other"),
        "AGE_BAND": age_band(df["AGE"]),
        "MARRIAGE": df["MARRIAGE"].map({1: "married", 2: "single", 3: "others"}).fillna("undocumented"),
    }


def build_operating_points(
    y_train,
    oof_prob,
    y_test,
    test_prob,
    capacities=(0.10, 0.20, 0.30),
    cost_ratios=(2, 3, 5, 10),
) -> list[dict]:
    """Menu of ways to turn probabilities into flags.

    Thresholds come from OUT-OF-FOLD TRAIN predictions only; the test set is used
    just to report what each setting would have achieved.
      * cost_ratio: minimise expected cost for "missing a default costs r x a false alarm"
      * capacity:   flag the riskiest X% of customers (the analyst team's workload)
    """
    oof_prob = np.asarray(oof_prob)
    y_test, test_prob = np.asarray(y_test), np.asarray(test_prob)
    default_ratio = config.COST_FALSE_NEGATIVE / config.COST_FALSE_POSITIVE
    ratios = sorted(set(cost_ratios) | {default_ratio})

    specs = []
    for r in ratios:
        t = choose_threshold(
            y_train, oof_prob,
            cost_fn=float(r) * config.COST_FALSE_POSITIVE, cost_fp=config.COST_FALSE_POSITIVE,
        )
        specs.append(("cost_ratio", f"cost_{r:g}", f"Cost ratio {r:g}:1", t, float(r), r == default_ratio))
    for c in capacities:
        t = float(np.clip(np.quantile(oof_prob, 1 - c), 0.01, 0.99))
        specs.append(("capacity", f"capacity_{int(round(c * 100))}", f"Review top {c:.0%}", t, None, False))

    positives = int((y_test == 1).sum())
    rows = []
    for kind, op_id, label, t, ratio, is_default in specs:
        pred = test_prob >= t
        tp = int((pred & (y_test == 1)).sum())
        fp = int((pred & (y_test == 0)).sum())
        rows.append(
            {
                "id": op_id,
                "kind": kind,
                "label": label,
                "cost_ratio": ratio,
                "threshold": float(t),
                "test_flag_rate": float(pred.mean()),
                "test_recall": tp / max(positives, 1),
                "test_precision": tp / max(tp + fp, 1),
                "is_default": bool(is_default),
            }
        )
    return sorted(rows, key=lambda d: d["test_flag_rate"])
