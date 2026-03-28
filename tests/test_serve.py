"""Tests for serve.py — FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from process_control_causal_ml.serve import app

# The TestClient triggers the lifespan; models load if data files exist.
# Tests are written to pass regardless of whether models are present.


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


VALID_READING = {
    "catalyst_type": "B",
    "coolant_flow_rate": 50.0,
    "reactor_temp": 183.0,
    "pressure": 2.75,
    "ph_level": 6.9,
    "reaction_rate": 74.0,
    "product_yield": 86.5,
}


def test_health_returns_200(client) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_response_schema(client) -> None:
    resp = client.get("/health")
    body = resp.json()
    assert "status" in body
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert isinstance(body["model_loaded"], bool)


def test_predict_invalid_body_returns_422(client) -> None:
    """Missing required fields should return 422 Unprocessable Entity."""
    resp = client.post("/predict", json={"reactor_temp": 183.0})
    assert resp.status_code == 422


def test_predict_invalid_ph_returns_422(client) -> None:
    """ph_level outside [0, 14] should be rejected by Pydantic validation."""
    bad = {**VALID_READING, "ph_level": 20.0}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_valid_body(client) -> None:
    """Valid body should return 200 or 503 (503 only if models are missing)."""
    resp = client.post("/predict", json=VALID_READING)
    assert resp.status_code in (200, 503)


def test_predict_200_response_schema(client) -> None:
    """If models are loaded, response must contain all expected fields."""
    resp = client.post("/predict", json=VALID_READING)
    if resp.status_code != 200:
        pytest.skip("Models not loaded in this environment")
    body = resp.json()
    assert "anomaly_flag" in body
    assert "anomaly_score" in body
    assert "anomaly_type" in body
    assert "corrective_action" in body
    action = body["corrective_action"]
    for field in ("variable", "current", "recommended", "delta", "confidence", "reasoning"):
        assert field in action


def test_predict_503_without_models(client, monkeypatch) -> None:
    """predict should return 503 when model_loaded is False."""
    import process_control_causal_ml.serve as serve_mod

    monkeypatch.setattr(serve_mod._state, "model_loaded", False)
    resp = client.post("/predict", json=VALID_READING)
    assert resp.status_code == 503


def test_causal_graph_missing_file_returns_404(client, monkeypatch, tmp_path) -> None:
    """causal_graph endpoint returns 404 when the PNG does not exist."""
    import process_control_causal_ml.serve as serve_mod

    monkeypatch.setattr(serve_mod, "DATA_DIR", tmp_path)
    resp = client.get("/causal_graph")
    assert resp.status_code == 404


def test_ate_endpoint_returns_200(client) -> None:
    resp = client.get("/ate")
    assert resp.status_code == 200


def test_ate_response_schema(client) -> None:
    body = client.get("/ate").json()
    assert "treatment" in body
    assert "outcome" in body
    assert "estimator" in body
    assert "ate" in body
