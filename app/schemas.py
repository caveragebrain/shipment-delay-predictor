"""Pydantic request/response schemas for the FastAPI app.

Field validation is intentionally forgiving on enum casing: clients can send
'F' / 'f', 'Road' / 'road', 'Low' / 'low' — we normalize to lowercase before
feeding the preprocessor (which expects lowercase, per src/preprocess.py).
"""

from __future__ import annotations

from typing import List, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Allowed values (lowercased internally — validators accept any case)
WAREHOUSES = {"a", "b", "c", "d", "f"}
MODES = {"ship", "flight", "road"}
IMPORTANCE = {"low", "medium", "high"}
GENDERS = {"m", "f"}


class ShipmentInput(BaseModel):
    warehouse_block: str = Field(..., description="Warehouse: A, B, C, D, or F")
    mode_of_shipment: str = Field(..., description="Ship, Flight, or Road")
    customer_care_calls: int = Field(..., ge=0, le=20)
    customer_rating: int = Field(..., ge=1, le=5)
    cost_of_product: float = Field(..., gt=0)
    prior_purchases: int = Field(..., ge=0, le=50)
    product_importance: str = Field(..., description="Low, Medium, or High")
    gender: str = Field(..., description="M or F")
    discount_offered: float = Field(..., ge=0, le=100)
    weight_in_gms: float = Field(..., gt=0)
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Decision threshold (0–1). If omitted, uses model_metadata default.",
    )

    @field_validator("warehouse_block")
    @classmethod
    def _v_warehouse(cls, v: str) -> str:
        s = str(v).strip().lower()
        if s not in WAREHOUSES:
            raise ValueError(f"warehouse_block must be one of {sorted(WAREHOUSES)}")
        return s

    @field_validator("mode_of_shipment")
    @classmethod
    def _v_mode(cls, v: str) -> str:
        s = str(v).strip().lower()
        if s not in MODES:
            raise ValueError(f"mode_of_shipment must be one of {sorted(MODES)}")
        return s

    @field_validator("product_importance")
    @classmethod
    def _v_importance(cls, v: str) -> str:
        s = str(v).strip().lower()
        if s not in IMPORTANCE:
            raise ValueError(f"product_importance must be one of {sorted(IMPORTANCE)}")
        return s

    @field_validator("gender")
    @classmethod
    def _v_gender(cls, v: str) -> str:
        s = str(v).strip().lower()
        if s not in GENDERS:
            raise ValueError(f"gender must be one of {sorted(GENDERS)}")
        return s


class PredictionResponse(BaseModel):
    delayed: bool
    probability: float
    threshold_used: float
    confidence: Literal["high", "medium", "low"]
    note: Optional[str] = None  # used by /worst-case, /best-case to describe inputs


class FactorOut(BaseModel):
    feature: str
    direction: Literal["increases_delay_risk", "decreases_delay_risk"]
    magnitude: float


class ExplainResponse(PredictionResponse):
    top_factors: List[FactorOut]
    explanation: str
    suggested_actions: List[str]


class FeatureRange(BaseModel):
    min_prob: float
    max_prob: float
    most_impactful_value: float
    sweep: List[Dict[str, float]]  # [{"value": x, "probability": p}, ...]


class SensitivityResponse(BaseModel):
    base_probability: float
    feature_ranges: Dict[str, FeatureRange]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    threshold_default: float
    gemini_configured: bool


def confidence_label(probability: float) -> Literal["high", "medium", "low"]:
    """Confidence band based on distance from 0.5."""
    if probability >= 0.75 or probability <= 0.25:
        return "high"
    if probability >= 0.65 or probability <= 0.35:
        return "medium"
    return "low"
