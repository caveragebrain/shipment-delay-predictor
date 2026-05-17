"""SHAP-based factor extraction + Gemini-powered natural-language explanations.

Two responsibilities, kept in the same module because they're always used together:
  1. `top_factors(...)` — turn a SHAP value vector into the K most-impactful
     human-readable contributors for a single prediction.
  2. `gemini_explain(...)` — call Gemini with a structured prompt and parse its
     JSON response into {explanation, suggested_actions}. If Gemini is unavailable
     or returns malformed JSON, degrade gracefully — never bring down the request.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")


def top_factors(
    shap_values: np.ndarray,
    feature_names: List[str],
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Return the top-K factors by absolute SHAP magnitude for one prediction.

    Each factor is {feature, direction, magnitude} where:
      - direction = "increases_delay_risk" if SHAP > 0 else "decreases_delay_risk"
        (positive SHAP pushes prediction toward class 1 = delayed)
      - magnitude = absolute SHAP value (raw, in log-odds units for XGBoost)
    """
    sv = np.asarray(shap_values).ravel()
    order = np.argsort(-np.abs(sv))[:k]
    return [
        {
            "feature": feature_names[i],
            "direction": "increases_delay_risk" if sv[i] > 0 else "decreases_delay_risk",
            "magnitude": float(abs(sv[i])),
            "signed_value": float(sv[i]),
        }
        for i in order
    ]


def _format_factors_for_prompt(factors: List[Dict[str, Any]]) -> str:
    lines = []
    for f in factors:
        arrow = "↑" if f["direction"] == "increases_delay_risk" else "↓"
        lines.append(f"  - {f['feature']:32s}  {arrow}  magnitude={f['magnitude']:.3f}")
    return "\n".join(lines)


def build_prompt(
    shipment: Dict[str, Any],
    probability: float,
    delayed: bool,
    threshold: float,
    factors: List[Dict[str, Any]],
) -> str:
    """Construct the Gemini prompt. Asks for JSON-only output."""
    verdict = "DELAYED" if delayed else "ON TIME"
    direction_word = "delayed" if delayed else "on time"
    return f"""You are a logistics operations analyst. A machine learning model has assessed a shipment.

SHIPMENT DETAILS:
- Mode: {shipment.get('mode_of_shipment')}
- Warehouse Block: {shipment.get('warehouse_block')}
- Weight: {shipment.get('weight_in_gms')}g
- Discount: {shipment.get('discount_offered')}%
- Customer care calls: {shipment.get('customer_care_calls')}
- Product importance: {shipment.get('product_importance')}
- Customer rating: {shipment.get('customer_rating')}
- Prior purchases: {shipment.get('prior_purchases')}
- Cost: ${shipment.get('cost_of_product')}

MODEL PREDICTION: {probability:.0%} probability of delay — classified as {verdict} (threshold: {threshold})

TOP CONTRIBUTING FACTORS (from SHAP analysis):
{_format_factors_for_prompt(factors)}

Task:
1. Write 2–3 sentences explaining WHY this shipment is predicted to be {direction_word},
   grounded in the contributing factors above. Be specific to the numbers.
2. Suggest exactly 3 concrete operational actions a logistics manager could take RIGHT NOW
   to reduce delay risk (or to maintain on-time delivery if already low risk).
   Each action should be one sentence, actionable, and specific to this shipment's profile.

Respond ONLY in this JSON format (no markdown, no code fences):
{{
  "explanation": "...",
  "suggested_actions": ["action 1", "action 2", "action 3"]
}}
"""


def _extract_json(text: str) -> Dict[str, Any] | None:
    """Best-effort JSON extraction. Gemini sometimes wraps JSON in ```json blocks."""
    if not text:
        return None
    # Strip code fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    # Find first balanced JSON object
    start = candidate.find("{")
    if start == -1:
        return None
    try:
        return json.loads(candidate[start:])
    except json.JSONDecodeError:
        # One more try: trim trailing junk after last }
        end = candidate.rfind("}")
        if end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def gemini_explain(
    shipment: Dict[str, Any],
    probability: float,
    delayed: bool,
    threshold: float,
    factors: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Call Gemini and return {"explanation": str, "suggested_actions": list[str]}.

    Always returns a dict (never raises). On any failure, returns a deterministic
    fallback explanation derived from the SHAP factors directly.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        return _fallback_explanation(factors, probability, delayed, reason="no_api_key")

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        prompt = build_prompt(shipment, probability, delayed, threshold, factors)
        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        parsed = _extract_json(text)
        if (
            parsed
            and isinstance(parsed.get("explanation"), str)
            and isinstance(parsed.get("suggested_actions"), list)
        ):
            # Trim to 3 actions, coerce all to strings
            actions = [str(a) for a in parsed["suggested_actions"][:3]]
            return {"explanation": parsed["explanation"], "suggested_actions": actions}
        logger.warning("Gemini returned unparseable response; using fallback.")
        return _fallback_explanation(
            factors, probability, delayed, reason="parse_failed", raw=text
        )
    except Exception as exc:
        logger.warning("Gemini call failed: %s", exc)
        return _fallback_explanation(factors, probability, delayed, reason=str(exc))


def _fallback_explanation(
    factors: List[Dict[str, Any]],
    probability: float,
    delayed: bool,
    reason: str = "",
    raw: str = "",
) -> Dict[str, Any]:
    """Deterministic explanation derived from SHAP factors. Used when Gemini fails."""
    verdict = "delayed" if delayed else "on time"
    top = factors[:3]
    pieces = []
    for f in top:
        verb = "raises" if f["direction"] == "increases_delay_risk" else "lowers"
        pieces.append(f"{f['feature']} {verb} risk")
    factors_text = "; ".join(pieces) if pieces else "no dominant factor"
    explanation = (
        f"Predicted {verdict} with {probability:.0%} probability of delay. "
        f"Largest contributors: {factors_text}. "
        f"(Automated explanation — LLM unavailable.)"
    )
    actions = [
        "Verify the shipment's current routing and ETA against the carrier's tracking feed.",
        "If risk is elevated, proactively notify the customer with a revised ETA window.",
        "Review the warehouse and shipping mode combination for known bottlenecks.",
    ]
    return {
        "explanation": explanation,
        "suggested_actions": actions,
        "_fallback_reason": reason,
    }
