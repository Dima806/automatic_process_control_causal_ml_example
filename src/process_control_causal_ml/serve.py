"""FastAPI REST API for the process control pipeline."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from process_control_causal_ml.control import CorrectiveAction, recommend_action
from process_control_causal_ml.detect import AnomalyDetector, AnomalyResult, detect_anomaly
from process_control_causal_ml.utils import load_config, logger

DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class AppState:
    causal_model: Any = None
    causal_model_config: Any = None
    detector: AnomalyDetector | None = None
    ate: float | None = None
    config: Any = None
    model_loaded: bool = False


_state = AppState()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ProcessReading(BaseModel):
    catalyst_type: str = Field(..., description="Catalyst type: A, B, or C")
    coolant_flow_rate: float = Field(..., ge=0, description="Coolant flow rate (L/min)")
    reactor_temp: float = Field(..., description="Reactor temperature (°C)")
    pressure: float = Field(..., ge=0, description="Reactor pressure (bar)")
    ph_level: float = Field(..., ge=0, le=14, description="pH level")
    reaction_rate: float = Field(..., description="Reaction rate (mol/s)")
    product_yield: float = Field(..., description="Product yield (%)")


class CorrectiveActionResponse(BaseModel):
    variable: str
    current: float
    recommended: float
    delta: float
    confidence: float
    reasoning: str


class PredictResponse(BaseModel):
    anomaly_flag: bool
    anomaly_score: float
    anomaly_type: str
    anomaly_variable: str
    corrective_action: CorrectiveActionResponse


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class CausalGraphResponse(BaseModel):
    image_base64: str


class ATEResponse(BaseModel):
    ate: float | None
    treatment: str
    outcome: str
    estimator: str


# ---------------------------------------------------------------------------
# Lifespan: load models at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load models at startup, release at shutdown."""
    logger.info("Loading models...")
    try:
        _state.config = load_config()

        # Load causal model
        model_path = DATA_DIR / "causal_model.pkl"
        if model_path.exists():
            saved = joblib.load(model_path)
            _state.causal_model = saved["model"]
            _state.causal_model_config = saved.get("config", _state.config.causal_model)
            logger.info("Causal model loaded")

        # Load detector
        detector_path = DATA_DIR / "detector.pkl"
        if detector_path.exists():
            _state.detector = joblib.load(detector_path)
            logger.info("Anomaly detector loaded")

        # Load ATE results
        ate_path = DATA_DIR / "ate_results.json"
        if ate_path.exists():
            with open(ate_path) as f:
                ate_data = json.load(f)
            _state.ate = ate_data.get("ate")

        _state.model_loaded = _state.causal_model is not None and _state.detector is not None
        logger.info(f"Startup complete. model_loaded={_state.model_loaded}")

    except Exception as exc:
        logger.error(f"Startup error: {exc}")

    yield

    logger.info("Shutting down...")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Process Control Causal ML API",
    description="Autonomous process control with causal ML",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok", model_loaded=_state.model_loaded)


@app.post("/predict", response_model=PredictResponse)
def predict(reading: ProcessReading) -> PredictResponse:
    """Detect anomalies and recommend corrective actions for a sensor reading."""
    if not _state.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run 'make train' first.",
        )

    logger.info(
        f"Predict request: temp={reading.reactor_temp:.1f}, yield={reading.product_yield:.1f}"
    )

    # Convert to dict for internal functions
    reading_dict = reading.model_dump()

    # Layer 1 & 2: Detect anomaly
    assert _state.detector is not None
    anomaly: AnomalyResult = detect_anomaly(reading_dict, _state.detector)

    # Recommend corrective action
    action: CorrectiveAction = recommend_action(
        anomaly=anomaly,
        causal_model=_state.causal_model,
        current_state=reading_dict,
        config=_state.config.control,  # type: ignore[union-attr]
    )

    return PredictResponse(
        anomaly_flag=anomaly.flag,
        anomaly_score=round(anomaly.score, 4),
        anomaly_type=anomaly.type,
        anomaly_variable=anomaly.variable,
        corrective_action=CorrectiveActionResponse(
            variable=action.variable,
            current=round(action.current, 4),
            recommended=round(action.recommended, 4),
            delta=round(action.delta, 4),
            confidence=round(action.confidence, 4),
            reasoning=action.reasoning,
        ),
    )


@app.get("/causal_graph", response_model=CausalGraphResponse)
def causal_graph() -> CausalGraphResponse:
    """Return base64-encoded PNG of the learned causal DAG."""
    graph_path = DATA_DIR / "causal_graph.png"
    if not graph_path.exists():
        raise HTTPException(
            status_code=404, detail="Causal graph not found. Run 'make graph' first."
        )

    with open(graph_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    return CausalGraphResponse(image_base64=image_b64)


@app.get("/ate", response_model=ATEResponse)
def ate() -> ATEResponse:
    """Return the Average Treatment Effect summary."""
    config = _state.config
    if config is None:
        config = load_config()

    ate_path = DATA_DIR / "ate_results.json"
    if ate_path.exists():
        with open(ate_path) as f:
            ate_data = json.load(f)
        return ATEResponse(
            ate=ate_data.get("ate"),
            treatment=ate_data.get("treatment", config.causal_model.treatment),
            outcome=ate_data.get("outcome", config.causal_model.outcome),
            estimator=ate_data.get("estimator", config.causal_model.estimator),
        )

    return ATEResponse(
        ate=_state.ate,
        treatment=config.causal_model.treatment,
        outcome=config.causal_model.outcome,
        estimator=config.causal_model.estimator,
    )
