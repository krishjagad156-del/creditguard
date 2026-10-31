"""Loads the trained pipeline ONCE and turns validated input into a prediction."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from creditguard import __version__, config


# Risk level compares a customer with the AVERAGE customer in the training data.
# The cut-points (0.5x, 1.5x, 3x the average default rate) are an ASSUMPTION for
# readability, not values learned from data. They do not depend on the operating point.
RISK_LEVELS = [(0.5, "Below average"), (1.5, "Around average"), (3.0, "Elevated")]


def relative_risk_band(prob: float, base_rate: float) -> str:
    ratio = prob / max(base_rate, 1e-9)
    for limit, label in RISK_LEVELS:
        if ratio < limit:
            return label
    return "Very high"


class ModelNotLoadedError(RuntimeError):
    pass


class ModelService:
    def __init__(self, model_path: Path, meta_path: Path):
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self.model = None
        self.meta: dict = {}

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not (self.model_path.exists() and self.meta_path.exists()):
            return  # stay "not loaded"; /health reports degraded, /predict returns 503
        self.model = joblib.load(self.model_path)
        self.meta = json.loads(self.meta_path.read_text())

    @property
    def version(self) -> str:
        return f"{__version__}+{self.meta.get('selected_model', 'unknown')}"

    @property
    def base_rate(self) -> float:
        return float(self.meta.get("base_rate") or self.meta["test_metrics"]["base_rate"])

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"])

    def predict(self, features: dict, threshold: float | None = None) -> dict:
        if not self.loaded:
            raise ModelNotLoadedError("Model artifacts are not available.")
        row = pd.DataFrame([features], columns=config.INPUT_COLUMNS)
        prob = float(self.model.predict_proba(row)[0, 1])
        t = self.threshold if threshold is None else float(threshold)
        return {
            "default_probability": prob,
            "risk_band": relative_risk_band(prob, self.base_rate),
            "average_default_rate": self.base_rate,
            "flagged_for_review": prob >= t,
            "threshold": t,
            "model_version": self.version,
        }
