"""Phase 3 experiments: do the engineered features and PAY one-hot encoding help?

    python -m creditguard.experiments

Only TRAIN data is used (5-fold stratified CV, same folds for every variant); the
test set is never touched. Because all variants see the same folds, we compare them
PER FOLD, which is more informative than comparing separate means and standard
deviations.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from creditguard import config
from creditguard.data import load_clean
from creditguard.features import build_preprocessor

REFERENCE = "engineered (current)"
VARIANTS = {
    "raw only": dict(engineer=False, pay_onehot=False),
    REFERENCE: dict(engineer=True, pay_onehot=False),
    "engineered + PAY one-hot": dict(engineer=True, pay_onehot=True),
}


def _models(seed: int) -> dict:
    return {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.05, max_iter=200, random_state=seed
        ),
    }


def run(df: pd.DataFrame | None = None, cv_folds: int = config.CV_FOLDS, out_dir: Path | None = None) -> pd.DataFrame:
    df = load_clean() if df is None else df
    seed = config.RANDOM_STATE
    train_df, _ = train_test_split(  # identical split to training; test half is discarded here
        df, test_size=config.TEST_SIZE, stratify=df[config.TARGET], random_state=seed
    )
    X, y = train_df[config.INPUT_COLUMNS], train_df[config.TARGET]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    rows = []
    for model_name, make_model in _models(seed).items():
        folds = {}
        for variant, kwargs in VARIANTS.items():
            pipe = Pipeline([("prep", build_preprocessor(**kwargs)), ("model", make_model())])
            res = cross_validate(
                pipe, X, y, cv=cv, scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"}
            )
            folds[variant] = res
        ref = folds[REFERENCE]["test_pr_auc"]
        for variant, res in folds.items():
            diff = res["test_pr_auc"] - ref
            rows.append(
                {
                    "model": model_name,
                    "variant": variant,
                    "cv_pr_auc": res["test_pr_auc"].mean(),
                    "cv_roc_auc": res["test_roc_auc"].mean(),
                    "pr_auc_diff_vs_current": diff.mean(),
                    "diff_std_across_folds": diff.std(),
                    "folds_better_than_current": int((diff > 0).sum()),
                    "n_folds": cv_folds,
                }
            )
    out = pd.DataFrame(rows)
    out_dir = Path(out_dir or config.PROJECT_ROOT / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "experiments.csv", index=False)
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    res = run()
    pd.options.display.float_format = "{:.4f}".format
    print("\nCross-validated results on the TRAIN set (test set untouched):\n")
    print(res.to_string(index=False))
    print("\nSaved reports/experiments.csv")


if __name__ == "__main__":
    main()
