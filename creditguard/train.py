"""Train, compare, evaluate and save the CreditGuard model.

Run from the project root:
    python -m creditguard.train                 # real data in data/raw/credit_default.csv
    python -m creditguard.train --tune          # adds a small hyper-parameter search
    python -m creditguard.train --synthetic     # smoke test only, NOT real results

Leakage controls:
  * the test split is carved off first and touched once, at the very end
  * every learned step (imputer, scaler, encoder, model) sits inside a Pipeline
    that is re-fit on each training fold
  * the decision threshold is chosen from out-of-fold TRAIN predictions
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from creditguard import __version__, config
from creditguard.data import clean_data, load_raw, standardize_columns
from creditguard.evaluation import (
    audit_groups,
    build_operating_points,
    calibration_table,
    choose_threshold,
    classification_report_at,
    fairness_audit,
    threshold_free_metrics,
)
from creditguard.features import build_preprocessor
from creditguard.synthetic import make_synthetic


def candidate_models(random_state: int) -> dict:
    """name -> (estimator, preprocessing options). Baseline first, then increasing complexity.

    The one-hot logistic variant exists because the EDA showed repayment status is not
    a straight line (status -1 defaults more than 0), and the experiments confirmed
    that a linear model benefits from treating it as categories.
    """
    return {
        "baseline_prior": (DummyClassifier(strategy="prior"), {}),
        "logistic_regression": (LogisticRegression(max_iter=1000), {}),
        "logistic_regression_pay_onehot": (LogisticRegression(max_iter=1000), {"pay_onehot": True}),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_leaf=20,
                n_jobs=-1, random_state=random_state,
            ),
            {},
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(
                max_depth=4, learning_rate=0.05, max_iter=200, random_state=random_state
            ),
            {},
        ),
    }


def make_pipeline(model, **prep_kwargs) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(**prep_kwargs)), ("model", model)])


def compare_models(X_train, y_train, cv, random_state: int) -> pd.DataFrame:
    rows = []
    for name, (model, prep) in candidate_models(random_state).items():
        res = cross_validate(
            make_pipeline(model, **prep), X_train, y_train, cv=cv,
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
        )
        rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": res["test_roc_auc"].mean(),
                "cv_roc_auc_std": res["test_roc_auc"].std(),
                "cv_pr_auc_mean": res["test_pr_auc"].mean(),
                "cv_pr_auc_std": res["test_pr_auc"].std(),
            }
        )
        print(f"  {name:<32} ROC-AUC {rows[-1]['cv_roc_auc_mean']:.4f}  PR-AUC {rows[-1]['cv_pr_auc_mean']:.4f}")
    return pd.DataFrame(rows).sort_values("cv_pr_auc_mean", ascending=False).reset_index(drop=True)


def tune_hgb(X_train, y_train, cv, random_state: int, n_iter: int = 12):
    search = RandomizedSearchCV(
        make_pipeline(HistGradientBoostingClassifier(random_state=random_state)),
        param_distributions={
            "model__max_depth": [3, 4, 5, 6],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_iter": [100, 200, 300],
            "model__min_samples_leaf": [20, 50, 100],
            "model__l2_regularization": [0.0, 0.1, 1.0],
        },
        n_iter=n_iter, scoring="average_precision", cv=cv,
        random_state=random_state, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"  tuned HGB best CV PR-AUC {search.best_score_:.4f} with {search.best_params_}")
    return search


def save_figures(model, X_test, y_test, y_prob, out_dir: Path, n_repeats: int = 5) -> dict:
    """Save ROC/PR, calibration and permutation-importance plots. Returns importances."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import CalibrationDisplay
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=axes[0])
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=axes[1])
    CalibrationDisplay.from_predictions(y_test, y_prob, n_bins=10, strategy="quantile", ax=axes[2])
    axes[0].set_title("ROC (test)"); axes[1].set_title("Precision-Recall (test)"); axes[2].set_title("Calibration (test)")
    fig.tight_layout(); fig.savefig(out_dir / "test_curves.png", dpi=130); plt.close(fig)

    perm = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=n_repeats,
        random_state=config.RANDOM_STATE, n_jobs=1,
    )
    imp = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.index, top.values)
    ax.set_title("Permutation importance on test set (drop in ROC-AUC)")
    fig.tight_layout(); fig.savefig(out_dir / "permutation_importance.png", dpi=130); plt.close(fig)
    return {k: float(v) for k, v in imp.items()}


def run(
    data_path: Path | None = None,
    synthetic: bool = False,
    tune: bool = False,
    model_path: Path | None = None,
    meta_path: Path | None = None,
    figures_dir: Path | None = None,
    cv_folds: int = config.CV_FOLDS,
    n_synthetic: int = 3000,
    make_figures: bool = True,
) -> dict:
    model_path = Path(model_path or config.MODEL_PATH)
    meta_path = Path(meta_path or config.META_PATH)
    figures_dir = Path(figures_dir or config.FIGURES_DIR)
    seed = config.RANDOM_STATE

    # ---- 1. load + clean -------------------------------------------------
    if synthetic:
        print("!! Using SYNTHETIC data. Metrics below are NOT real results.")
        df = clean_data(standardize_columns(make_synthetic(n_synthetic)))
    else:
        df = clean_data(load_raw(data_path))
    print(f"Loaded {len(df):,} rows; default rate {df[config.TARGET].mean():.3f}")

    # ---- 2. split FIRST (test set is locked away) -------------------------
    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df[config.TARGET], random_state=seed
    )
    X_train, y_train = train_df[config.INPUT_COLUMNS], train_df[config.TARGET]
    X_test, y_test = test_df[config.INPUT_COLUMNS], test_df[config.TARGET]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # ---- 3. compare models with cross-validation on TRAIN only -----------
    print("Cross-validated comparison (train set only):")
    comparison = compare_models(X_train, y_train, cv, seed)
    best_name = comparison.loc[comparison["model"] != "baseline_prior"].iloc[0]["model"]
    best_params = {}
    best_model, best_prep = candidate_models(seed)[best_name]
    if tune and best_name == "hist_gradient_boosting":
        search = tune_hgb(X_train, y_train, cv, seed)
        best_model = search.best_estimator_.named_steps["model"]
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    print(f"Selected model: {best_name}")

    # ---- 4. calibrated final model; threshold from OOF train predictions --
    final = CalibratedClassifierCV(make_pipeline(best_model, **best_prep), method="sigmoid", cv=cv_folds)
    oof_prob = cross_val_predict(final, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = choose_threshold(y_train, oof_prob)
    print(f"Cost-aware threshold from out-of-fold train predictions: {threshold:.2f}")

    final.fit(X_train, y_train)

    # ---- 5. single, final evaluation on the held-out test set -------------
    test_prob = final.predict_proba(X_test)[:, 1]
    test_metrics = {**threshold_free_metrics(y_test, test_prob), **classification_report_at(y_test, test_prob, threshold)}
    baseline_cost = classification_report_at(y_test, np.zeros(len(y_test)), 0.5)["expected_cost_per_customer"]
    test_metrics["cost_if_nobody_flagged"] = baseline_cost
    print("Test metrics:", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in test_metrics.items()}))

    operating_points = build_operating_points(y_train, oof_prob, y_test, test_prob)
    print("Operating points (thresholds from train OOF, results on test):")
    for op in operating_points:
        print(f"  {op['label']:<18} threshold {op['threshold']:.3f}  flags {op['test_flag_rate']:.1%}  "
              f"recall {op['test_recall']:.1%}  precision {op['test_precision']:.1%}")

    fairness = {
        attr: fairness_audit(y_test, test_prob, labels.loc[X_test.index], threshold)
        for attr, labels in audit_groups(test_df).items()
    }

    importances = {}
    if make_figures:
        importances = save_figures(final, X_test, y_test, test_prob, figures_dir)

    # ---- 6. persist the full fitted pipeline + metadata -------------------
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, model_path)
    meta = {
        "project_version": __version__,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trained_on_synthetic_data": bool(synthetic),
        "selected_model": best_name,
        "tuned_params": best_params,
        "preprocessing": best_prep,
        "input_columns": config.INPUT_COLUMNS,
        "audit_only_columns": config.AUDIT_COLUMNS,
        "threshold": threshold,
        "cost_assumptions": {
            "false_negative": config.COST_FALSE_NEGATIVE,
            "false_positive": config.COST_FALSE_POSITIVE,
        },
        "split": {"test_size": config.TEST_SIZE, "random_state": seed, "n_train": len(train_df), "n_test": len(test_df)},
        "cv_comparison": comparison.round(5).to_dict(orient="records"),
        "test_metrics": test_metrics,
        "base_rate": float(y_train.mean()),
        "operating_points": operating_points,
        "calibration": calibration_table(y_test, test_prob),
        "fairness_audit": fairness,
        "permutation_importance": importances,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved model -> {model_path}\nSaved metadata -> {meta_path}")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=None, help="path to the CSV (default: data/raw/credit_default.csv)")
    parser.add_argument("--tune", action="store_true", help="small randomized search for gradient boosting")
    parser.add_argument("--synthetic", action="store_true", help="smoke test on synthetic data (NOT real results)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    run(data_path=args.data, synthetic=args.synthetic, tune=args.tune, make_figures=not args.no_figures)


if __name__ == "__main__":
    main()
