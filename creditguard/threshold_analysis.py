"""Threshold sensitivity analysis.

The model's flag threshold depends on an ASSUMED cost ratio (missing a defaulter vs
reviewing a good customer). This script shows how recall, precision and cost change
if (a) the analyst team can only review a fixed share of customers, or (b) the cost
ratio is different from the assumed one.

    python -m creditguard.threshold_analysis

Leakage rule: thresholds are always chosen from OUT-OF-FOLD TRAIN predictions and
only then applied to the test set, exactly like in training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from creditguard import config
from creditguard.data import load_clean
from creditguard.evaluation import choose_threshold, classification_report_at

CAPACITIES = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
COST_RATIOS = [1, 2, 3, 5, 10, 20]


def capacity_table(y_true, y_prob, fractions=CAPACITIES) -> pd.DataFrame:
    """If only the riskiest `fraction` of customers can be reviewed."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    base = y_true.mean()
    rows = []
    for f in fractions:
        cut = np.quantile(y_prob, 1 - f)
        flagged = y_prob >= cut
        tp = int((flagged & (y_true == 1)).sum())
        rows.append(
            {
                "review_share": f,
                "prob_cutoff": float(cut),
                "defaulters_caught": tp,
                "recall": tp / max(int((y_true == 1).sum()), 1),
                "precision": tp / max(int(flagged.sum()), 1),
                "lift_vs_random": (tp / max(int(flagged.sum()), 1)) / base,
            }
        )
    return pd.DataFrame(rows)


def cost_ratio_table(y_train, oof_prob, y_test, test_prob, ratios=COST_RATIOS) -> pd.DataFrame:
    """For each assumed cost ratio: threshold from train OOF, results on test."""
    rows = []
    for r in ratios:
        t = choose_threshold(y_train, oof_prob, cost_fn=float(r), cost_fp=1.0)
        pred = (np.asarray(test_prob) >= t).astype(int)
        tp = int(((pred == 1) & (np.asarray(y_test) == 1)).sum())
        fn = int(((pred == 0) & (np.asarray(y_test) == 1)).sum())
        fp = int(((pred == 1) & (np.asarray(y_test) == 0)).sum())
        n = len(pred)
        rows.append(
            {
                "cost_ratio_FN_to_FP": r,
                "threshold": t,
                "flag_rate": float(pred.mean()),
                "recall": tp / max(tp + fn, 1),
                "precision": tp / max(tp + fp, 1),
                "cost_per_customer": (fn * r + fp * 1.0) / n,
                "cost_if_nobody_flagged": (int(np.sum(y_test)) * r) / n,
            }
        )
    return pd.DataFrame(rows)


def run(model_path: Path | None = None, df: pd.DataFrame | None = None,
        out_dir: Path | None = None, cv_folds: int = config.CV_FOLDS) -> dict:
    model = joblib.load(Path(model_path or config.MODEL_PATH))
    df = load_clean() if df is None else df

    train_df, test_df = train_test_split(  # identical split to training
        df, test_size=config.TEST_SIZE, stratify=df[config.TARGET], random_state=config.RANDOM_STATE
    )
    X_train, y_train = train_df[config.INPUT_COLUMNS], train_df[config.TARGET]
    X_test, y_test = test_df[config.INPUT_COLUMNS], test_df[config.TARGET]

    test_prob = model.predict_proba(X_test)[:, 1]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)
    oof = cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    cap = capacity_table(y_test, test_prob)
    ratios = cost_ratio_table(y_train, oof, y_test, test_prob)

    out_dir = Path(out_dir or config.PROJECT_ROOT / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    cap.to_csv(out_dir / "capacity_table.csv", index=False)
    ratios.to_csv(out_dir / "cost_ratio_table.csv", index=False)
    return {"capacity": cap, "cost_ratio": ratios}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    res = run()
    pd.options.display.float_format = "{:.3f}".format
    print("\nA) If only the riskiest X% of customers can be reviewed (test set):")
    print(res["capacity"].to_string(index=False))
    print("\nB) If the assumed cost ratio were different (threshold from train OOF, results on test):")
    print(res["cost_ratio"].to_string(index=False))
    print("\nSaved CSVs to reports/capacity_table.csv and reports/cost_ratio_table.csv")


if __name__ == "__main__":
    main()
