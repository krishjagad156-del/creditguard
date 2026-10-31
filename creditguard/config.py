"""Central configuration.

Paths and cost assumptions live here so nothing is hard-coded elsewhere.
Paths can be overridden with environment variables (see .env.example).
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = Path(os.getenv("DATA_PATH", PROJECT_ROOT / "data" / "raw" / "credit_default.csv"))
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "artifacts" / "model.joblib"))
META_PATH = Path(os.getenv("META_PATH", PROJECT_ROOT / "artifacts" / "model_meta.json"))
FIGURES_DIR = Path(os.getenv("FIGURES_DIR", PROJECT_ROOT / "reports" / "figures"))

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET = "default"

# Columns used ONLY for the fairness audit, never as model inputs.
AUDIT_COLUMNS = ["SEX", "AGE", "MARRIAGE"]

PAY_COLS = [f"PAY_{i}" for i in range(1, 7)]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]

# Raw columns the model (and the API) accept as input.
INPUT_COLUMNS = ["LIMIT_BAL", "EDUCATION", *PAY_COLS, *BILL_COLS, *PAYAMT_COLS]

# Cost ASSUMPTIONS for threshold selection. These are not measured from data:
# they encode "missing a defaulter is worse than reviewing a good customer".
# Change them to match a different business case and retrain.
COST_FALSE_NEGATIVE = 5.0
COST_FALSE_POSITIVE = 1.0
