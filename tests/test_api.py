"""Endpoint smoke tests using httpx's AsyncClient against the FastAPI app.

These tests boot the lifespan handler so the model is actually loaded — no mocks
on the model side. The Gemini call is monkeypatched to a deterministic stub so
tests don't hit the network and don't require GEMINI_API_KEY to be set in CI.

Run from the repo root:
    pytest -q
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from src import explain as explain_module

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE = {
    "warehouse_block": "F",
    "mode_of_shipment": "Road",
    "customer_care_calls": 4,
    "customer_rating": 2,
    "cost_of_product": 180.0,
    "prior_purchases": 3,
    "product_importance": "High",
    "gender": "M",
    "discount_offered": 45.0,
    "weight_in_gms": 4200.0,
}


@pytest.fixture(autouse=True)
def _stub_gemini(monkeypatch):
    """Replace the Gemini call with a deterministic stub so tests are hermetic."""
    def _fake(shipment, probability, delayed, threshold, factors):
        return {
            "explanation": f"TEST explanation: prob={probability:.2f} delayed={delayed}",
            "suggested_actions": ["test action 1", "test action 2", "test action 3"],
        }
    monkeypatch.setattr(explain_module, "gemini_explain", _fake)
    # Also patch the symbol re-imported into app.main
    import app.main as main_module
    monkeypatch.setattr(main_module, "gemini_explain", _fake)


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # Trigger lifespan startup explicitly
        async with app.router.lifespan_context(app):
            yield ac


@pytest.mark.asyncio
async def test_health_returns_ok(client):
    r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert 0 <= data["threshold_default"] <= 1


@pytest.mark.asyncio
async def test_predict_json_body_valid(client):
    r = await client.post("/predict", json=SAMPLE)
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data["delayed"], bool)
    assert 0.0 <= data["probability"] <= 1.0
    assert data["threshold_used"] > 0
    assert data["confidence"] in {"low", "medium", "high"}


@pytest.mark.asyncio
async def test_predict_respects_custom_threshold(client):
    high_t = dict(SAMPLE); high_t["threshold"] = 0.99
    r = await client.post("/predict", json=high_t)
    assert r.status_code == 200
    assert r.json()["threshold_used"] == 0.99


@pytest.mark.asyncio
async def test_predict_invalid_warehouse_returns_422(client):
    bad = dict(SAMPLE); bad["warehouse_block"] = "Z"
    r = await client.post("/predict", json=bad)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_predict_file_upload_single(client):
    body = json.dumps(SAMPLE).encode()
    files = {"file": ("shipment.json", io.BytesIO(body), "application/json")}
    r = await client.post("/predict", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "probability" in data


@pytest.mark.asyncio
async def test_predict_file_upload_batch(client):
    body = json.dumps([SAMPLE, SAMPLE]).encode()
    files = {"file": ("shipments.json", io.BytesIO(body), "application/json")}
    r = await client.post("/predict", files=files)
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data, list) and len(data) == 2


@pytest.mark.asyncio
async def test_explain_returns_factors_and_actions(client):
    r = await client.post("/explain", json=SAMPLE)
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["top_factors"]) == 5
    for f in data["top_factors"]:
        assert f["direction"] in {"increases_delay_risk", "decreases_delay_risk"}
    assert data["explanation"].startswith("TEST explanation")
    assert len(data["suggested_actions"]) == 3


@pytest.mark.asyncio
async def test_worst_case_returns_high_probability(client):
    r = await client.post("/worst-case")
    assert r.status_code == 200
    d = r.json()
    assert d["probability"] > 0.7
    assert d["note"]


@pytest.mark.asyncio
async def test_best_case_is_lower_than_worst(client):
    worst = (await client.post("/worst-case")).json()["probability"]
    best = (await client.post("/best-case")).json()["probability"]
    assert best < worst


@pytest.mark.asyncio
async def test_sensitivity_returns_all_numeric_features(client):
    r = await client.post("/sensitivity", json=SAMPLE)
    assert r.status_code == 200
    d = r.json()
    expected = {
        "customer_care_calls", "customer_rating", "cost_of_product",
        "prior_purchases", "discount_offered", "weight_in_gms",
    }
    assert set(d["feature_ranges"].keys()) == expected
    for feat, r_ in d["feature_ranges"].items():
        assert r_["max_prob"] >= r_["min_prob"]


@pytest.mark.asyncio
async def test_dataset_sample_returns_rows(client):
    r = await client.get("/dataset-sample?n=50")
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list) and 10 <= len(rows) <= 50
    assert "delayed" in rows[0]


@pytest.mark.asyncio
async def test_root_redirects_to_demo(client):
    r = await client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"].endswith("/demo")
