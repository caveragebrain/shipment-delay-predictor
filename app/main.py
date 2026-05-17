"""FastAPI app for the shipment-delay predictor.

Endpoints:
  GET  /                  -> redirect to /demo
  GET  /demo              -> interactive HTML demo
  GET  /health            -> liveness check + model status
  GET  /dataset-sample    -> 300 sampled raw rows for parallel-coordinates viz
  POST /predict           -> JSON body OR uploaded JSON file (single or batch)
  POST /explain           -> single-input SHAP + Gemini explanation
  POST /worst-case        -> canonical worst-case inputs, returns prediction
  POST /best-case         -> canonical best-case inputs, returns prediction
  POST /sensitivity       -> per-feature probability sweep for one shipment
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Union

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.predictor import (
    BEST_CASE_INPUT,
    PROCESSED_DIR,
    Predictor,
    WORST_CASE_INPUT,
)
from app.schemas import (
    ExplainResponse,
    FactorOut,
    HealthResponse,
    PredictionResponse,
    SensitivityResponse,
    ShipmentInput,
    confidence_label,
)
from src.explain import gemini_explain, top_factors

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("app")

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"


# --- Lifespan: load model once at startup -------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifacts ...")
    app.state.predictor = Predictor.load()
    app.state.gemini_configured = bool(os.getenv("GEMINI_API_KEY", "").strip())
    logger.info("Model loaded. Gemini configured: %s", app.state.gemini_configured)
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Shipment Delay Predictor",
    description="Predicts whether an e-commerce shipment will be delayed.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static files at /static (the demo page references /static/* assets if any)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --- Request logging middleware -----------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        '%s %s -> %d  (%.1f ms)',
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


# --- Helpers ------------------------------------------------------------------

def _resolved_threshold(p: Predictor, requested: float | None) -> float:
    return float(requested) if requested is not None else p.default_threshold


def _build_prediction_response(
    p: Predictor, shipment: dict, *, note: str | None = None
) -> PredictionResponse:
    threshold = _resolved_threshold(p, shipment.get("threshold"))
    proba = p.predict_proba(shipment)
    return PredictionResponse(
        delayed=bool(proba >= threshold),
        probability=round(proba, 4),
        threshold_used=round(threshold, 3),
        confidence=confidence_label(proba),
        note=note,
    )


# --- Routes -------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/demo")


@app.get("/demo", include_in_schema=False)
async def demo_page():
    html_path = STATIC_DIR / "demo.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="demo.html not found")
    return FileResponse(html_path)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    p: Predictor = request.app.state.predictor
    return HealthResponse(
        status="ok",
        model_loaded=p.model is not None,
        threshold_default=p.default_threshold,
        gemini_configured=request.app.state.gemini_configured,
    )


@app.get("/dataset-sample")
async def dataset_sample(n: int = 300):
    """Return up to N randomly sampled raw training rows for visualization.

    Reads from data/processed/raw_train.csv. The endpoint is read-only and
    purely informational — used by the demo's parallel-coordinates plot.
    """
    n = max(10, min(int(n), 2000))
    csv_path = PROCESSED_DIR / "raw_train.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="raw_train.csv missing; run preprocess first")
    df = pd.read_csv(csv_path)
    if len(df) > n:
        df = df.sample(n=n, random_state=42)
    return JSONResponse(df.to_dict(orient="records"))


@app.post("/predict")
async def predict(request: Request) -> Any:
    """Single prediction via JSON body, OR single/batch via uploaded JSON file.

    Dispatches on Content-Type:
      - application/json            -> single ShipmentInput in body
      - multipart/form-data + file  -> JSON file (object = single, array = batch)
    """
    p: Predictor = request.app.state.predictor
    content_type = (request.headers.get("content-type") or "").lower()

    # --- File upload path -----------------------------------------------------
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="Form field 'file' is required.")
        raw = await upload.read()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in file: {e}")

        if isinstance(data, list):
            try:
                shipments = [ShipmentInput(**row).model_dump() for row in data]
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Validation error in batch: {e}")
            return [_build_prediction_response(p, s) for s in shipments]
        if isinstance(data, dict):
            try:
                shipment = ShipmentInput(**data).model_dump()
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Validation error: {e}")
            return _build_prediction_response(p, shipment)
        raise HTTPException(status_code=400, detail="File must contain a JSON object or array.")

    # --- JSON body path -------------------------------------------------------
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Expected JSON body or multipart file: {e}")

    try:
        shipment = ShipmentInput(**body).model_dump()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Validation error: {e}")
    return _build_prediction_response(p, shipment)


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: Request, payload: ShipmentInput) -> ExplainResponse:
    p: Predictor = request.app.state.predictor
    shipment = payload.model_dump()
    threshold = _resolved_threshold(p, shipment.get("threshold"))
    proba = p.predict_proba(shipment)
    delayed = bool(proba >= threshold)

    shap_vec = p.shap_for(shipment)
    factors = top_factors(shap_vec, p.feature_names, k=5)

    # Gemini explanation (graceful fallback inside)
    gem = gemini_explain(shipment, proba, delayed, threshold, factors)

    return ExplainResponse(
        delayed=delayed,
        probability=round(proba, 4),
        threshold_used=round(threshold, 3),
        confidence=confidence_label(proba),
        top_factors=[FactorOut(**f) for f in factors],
        explanation=gem["explanation"],
        suggested_actions=gem["suggested_actions"],
    )


@app.post("/worst-case", response_model=PredictionResponse)
async def worst_case(request: Request) -> PredictionResponse:
    p: Predictor = request.app.state.predictor
    note = (
        "Worst-case inputs: warehouse F, road shipping, 65% discount, 7000g, "
        "high importance, low rating, minimum prior purchases, 7 care calls."
    )
    return _build_prediction_response(p, dict(WORST_CASE_INPUT), note=note)


@app.post("/best-case", response_model=PredictionResponse)
async def best_case(request: Request) -> PredictionResponse:
    p: Predictor = request.app.state.predictor
    note = (
        "Best-case inputs: warehouse A, flight, 5% discount, moderate weight, "
        "low importance, top customer rating, repeat buyer, minimal care calls."
    )
    return _build_prediction_response(p, dict(BEST_CASE_INPUT), note=note)


@app.post("/sensitivity", response_model=SensitivityResponse)
async def sensitivity(request: Request, payload: ShipmentInput) -> SensitivityResponse:
    p: Predictor = request.app.state.predictor
    result = p.sensitivity(payload.model_dump(), steps=10)
    return SensitivityResponse(**result)


# Generic exception handler so client always gets JSON
@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})
