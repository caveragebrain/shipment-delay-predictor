"""Model loading and inference orchestration.

The `Predictor` is constructed once at app startup (FastAPI lifespan) and
holds the fitted preprocessor, the trained XGBoost model, the SHAP explainer,
and model metadata. All inference paths route through this object so the API
layer stays focused on HTTP concerns.

SHAP explainer is rebuilt from the trained model on startup (no .pkl) — see
the earlier design decision to avoid SHAP-version pickling fragility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Repo paths (relative to this file)
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "model"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

REQUIRED_ARTIFACTS = [
    MODEL_DIR / "model.pkl",
    MODEL_DIR / "preprocessor.pkl",
    MODEL_DIR / "model_metadata.json",
]

# Observed numeric ranges in the training set, used for sensitivity sweeps and
# worst/best-case scenarios. These are NOT trained — they're domain knowledge
# encoded as constants so the API doesn't need the raw dataset at runtime.
NUMERIC_RANGES: Dict[str, tuple[float, float]] = {
    "customer_care_calls": (1, 7),
    "customer_rating": (1, 5),
    "cost_of_product": (96, 310),
    "prior_purchases": (2, 10),
    "discount_offered": (0, 65),
    "weight_in_gms": (1000, 7846),
}

# Worst-case / best-case canonical inputs (per spec)
WORST_CASE_INPUT: Dict[str, Any] = {
    "warehouse_block": "f",
    "mode_of_shipment": "road",
    "customer_care_calls": 7,
    "customer_rating": 1,
    "cost_of_product": 180.0,
    "prior_purchases": 2,
    "product_importance": "high",
    "gender": "m",
    "discount_offered": 65.0,
    "weight_in_gms": 7000.0,
}
BEST_CASE_INPUT: Dict[str, Any] = {
    "warehouse_block": "a",
    "mode_of_shipment": "flight",
    "customer_care_calls": 2,
    "customer_rating": 5,
    "cost_of_product": 180.0,
    "prior_purchases": 8,
    "product_importance": "low",
    "gender": "f",
    "discount_offered": 5.0,
    "weight_in_gms": 4500.0,
}


class Predictor:
    """Holds all model artifacts and exposes inference methods."""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.metadata: Dict[str, Any] = {}
        self.explainer = None
        self.feature_names: List[str] = []
        self.default_threshold: float = 0.5

    @classmethod
    def load(cls) -> "Predictor":
        missing = [str(p) for p in REQUIRED_ARTIFACTS if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing model artifacts: {missing}. Run `python -m src.train` first."
            )
        p = cls()
        p.model = joblib.load(MODEL_DIR / "model.pkl")
        p.preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
        with open(MODEL_DIR / "model_metadata.json") as f:
            p.metadata = json.load(f)
        p.feature_names = list(p.metadata.get("feature_names", []))
        p.default_threshold = float(p.metadata.get("threshold", 0.5))

        # Rebuild SHAP explainer from the trained booster (avoids pickle fragility)
        import shap

        p.explainer = shap.TreeExplainer(p.model)
        logger.info(
            "Predictor loaded: n_features=%d, default_threshold=%.2f",
            len(p.feature_names),
            p.default_threshold,
        )
        return p

    # --- core inference -----------------------------------------------------

    def _to_dataframe(self, shipment: Dict[str, Any]) -> pd.DataFrame:
        """Drop the API-only `threshold` field and wrap the rest in a DataFrame."""
        clean = {k: v for k, v in shipment.items() if k != "threshold"}
        return pd.DataFrame([clean])

    def _transform(self, shipment: Dict[str, Any]) -> np.ndarray:
        return self.preprocessor.transform(self._to_dataframe(shipment))

    def predict_proba(self, shipment: Dict[str, Any]) -> float:
        X = self._transform(shipment)
        return float(self.model.predict_proba(X)[0, 1])

    def predict_proba_batch(self, X_raw: pd.DataFrame) -> np.ndarray:
        X = self.preprocessor.transform(X_raw)
        return self.model.predict_proba(X)[:, 1]

    def shap_for(self, shipment: Dict[str, Any]) -> np.ndarray:
        """Return SHAP values for the (transformed) feature vector."""
        X = self._transform(shipment)
        # TreeExplainer for binary XGBClassifier returns shape (1, n_features)
        sv = self.explainer.shap_values(X)
        return np.asarray(sv).reshape(-1)

    # --- sensitivity --------------------------------------------------------

    def sensitivity(self, shipment: Dict[str, Any], steps: int = 10) -> Dict[str, Any]:
        """Sweep each numeric feature across its observed range, holding others fixed.

        Returns the base probability plus, per feature, the min/max probability seen
        across the sweep, the input value that produced the max, and the full sweep
        for visualization.
        """
        base_prob = self.predict_proba(shipment)
        out: Dict[str, Any] = {"base_probability": base_prob, "feature_ranges": {}}

        for feat, (lo, hi) in NUMERIC_RANGES.items():
            values = np.linspace(lo, hi, steps)
            sweep = []
            for v in values:
                modified = dict(shipment)
                modified[feat] = float(v)
                p = self.predict_proba(modified)
                sweep.append({"value": float(v), "probability": float(p)})
            probs = [s["probability"] for s in sweep]
            idx_max = int(np.argmax(probs))
            out["feature_ranges"][feat] = {
                "min_prob": float(min(probs)),
                "max_prob": float(max(probs)),
                "most_impactful_value": float(sweep[idx_max]["value"]),
                "sweep": sweep,
            }
        return out
