"""Error analysis: where does the model go wrong?

    python -m creditguard.error_analysis

Uses the saved model, the same test split as training, and the model's default
threshold. This is DESCRIPTIVE: it never changes the model, so looking at the test
set here does not leak anything into training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from creditguard import config
from creditguard.data import load_clean

PAY_ORDER = ["-2", "-1", "0", "1", "2", "3+"]
LIMIT_LABELS = ["Q1 lowest limit", "Q2", "Q3", "Q4", "Q5 highest limit"]


def segment_table(y, prob, flag, segment) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"segment": list(segment), "y": np.asarray(y), "p": np.asarray(prob), "flag": np.asarray(flag)}
    )
    order = list(segment.cat.categories) if hasattr(segment, "cat") else sorted(frame["segment"].unique())
    rows = []
    for seg in order:
        g = frame[frame["segment"] == seg]
        if g.empty:
            continue
        pos = int((g["y"] == 1).sum())
        tp = int(((g["flag"] == 1) & (g["y"] == 1)).sum())
        fp = int(((g["flag"] == 1) & (g["y"] == 0)).sum())
        rows.append(
            {
                "segment": str(seg),
                "n": len(g),
                "default_rate": g["y"].mean(),
                "mean_predicted": g["p"].mean(),
                "flag_rate": g["flag"].mean(),
                "recall": tp / pos if pos else np.nan,
                "precision": tp / (tp + fp) if (tp + fp) else np.nan,
                "missed_defaulters": pos - tp,
            }
        )
    return pd.DataFrame(rows)


def build_segments(X: pd.DataFrame) -> dict[str, pd.Series]:
    pay = np.where(X["PAY_1"] >= 3, "3+", X["PAY_1"].astype(int).astype(str))
    inactive = (X[config.BILL_COLS] == 0).all(axis=1) & (X[config.PAYAMT_COLS] == 0).all(axis=1)
    return {
        "Most recent repayment status (PAY_1)": pd.Series(
            pd.Categorical(pay, categories=PAY_ORDER, ordered=True), index=X.index
        ),
        "Credit limit quintile": pd.Series(
            pd.qcut(X["LIMIT_BAL"].rank(method="first"), 5, labels=LIMIT_LABELS), index=X.index
        ),
        "Account activity": pd.Series(
            pd.Categorical(np.where(inactive, "no bills, no payments", "active"),
                           categories=["active", "no bills, no payments"]), index=X.index
        ),
        "Any negative bill (credit balance)": pd.Series(
            pd.Categorical(np.where((X[config.BILL_COLS] < 0).any(axis=1), "yes", "no"),
                           categories=["no", "yes"]), index=X.index
        ),
    }


def run(model_path: Path | None = None, meta_path: Path | None = None,
        df: pd.DataFrame | None = None, out_dir: Path | None = None, threshold: float | None = None) -> dict:
    model = joblib.load(Path(model_path or config.MODEL_PATH))
    meta = json.loads(Path(meta_path or config.META_PATH).read_text())
    t = float(meta["threshold"] if threshold is None else threshold)
    df = load_clean() if df is None else df

    _, test_df = train_test_split(  # identical split to training
        df, test_size=config.TEST_SIZE, stratify=df[config.TARGET], random_state=config.RANDOM_STATE
    )
    X, y = test_df[config.INPUT_COLUMNS], test_df[config.TARGET]
    prob = model.predict_proba(X)[:, 1]
    flag = (prob >= t).astype(int)

    tables = {name: segment_table(y, prob, flag, seg) for name, seg in build_segments(X).items()}

    missed = X.assign(probability=prob, actual_default=y.values, flagged=flag)
    missed = missed[(missed["actual_default"] == 1) & (missed["flagged"] == 0)]
    hidden_share = float((missed["PAY_1"] <= 0).mean()) if len(missed) else float("nan")
    worst_misses = missed.sort_values("probability").head(10)[
        ["probability", "LIMIT_BAL", "PAY_1", "PAY_2", "PAY_3", "BILL_AMT1", "PAY_AMT1", "EDUCATION"]
    ]
    false_alarms = X.assign(probability=prob, actual_default=y.values, flagged=flag)
    false_alarms = false_alarms[(false_alarms["actual_default"] == 0) & (false_alarms["flagged"] == 1)]
    worst_alarms = false_alarms.sort_values("probability", ascending=False).head(10)[
        ["probability", "LIMIT_BAL", "PAY_1", "PAY_2", "PAY_3", "BILL_AMT1", "PAY_AMT1", "EDUCATION"]
    ]

    out_dir = Path(out_dir or config.PROJECT_ROOT / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat({k: v for k, v in tables.items()}, names=["slice"]).reset_index(level=0).to_csv(
        out_dir / "error_analysis_segments.csv", index=False
    )
    worst_misses.to_csv(out_dir / "error_analysis_worst_misses.csv", index=False)
    worst_alarms.to_csv(out_dir / "error_analysis_worst_false_alarms.csv", index=False)
    return {
        "threshold": t,
        "tables": tables,
        "n_missed": int(len(missed)),
        "missed_with_current_or_early_status": hidden_share,
        "worst_misses": worst_misses,
        "worst_false_alarms": worst_alarms,
    }


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    res = run()
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 200
    print(f"Threshold used: {res['threshold']:.2f} (the model's default operating point)\n")
    for name, table in res["tables"].items():
        print(f"== {name} ==")
        print(table.to_string(index=False), "\n")
    print(f"Missed defaulters: {res['n_missed']}; share with PAY_1 <= 0 (looked current): "
          f"{res['missed_with_current_or_early_status']:.1%}\n")
    print("== 10 defaulters the model was most sure were safe ==")
    print(res["worst_misses"].to_string(index=False), "\n")
    print("== 10 non-defaulters the model was most sure would default ==")
    print(res["worst_false_alarms"].to_string(index=False))
    print("\nSaved CSVs to reports/error_analysis_*.csv")


if __name__ == "__main__":
    main()
