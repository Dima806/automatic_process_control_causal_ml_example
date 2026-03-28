"""Configuration models and shared utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

logger.remove()
logger.add(
    sys.stderr,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} - {message}",
    level="INFO",
)


# ---------------------------------------------------------------------------
# Pydantic config models
# ---------------------------------------------------------------------------


class SimulationConfig(BaseModel):
    n_batches: int = 50000
    anomaly_fraction: float = 0.05
    random_seed: int = 42
    catalyst_types: list[str] = ["A", "B", "C"]
    catalyst_temp_effects: dict[str, float] = {"A": 0.0, "B": 2.5, "C": -1.5}


class CausalGraphConfig(BaseModel):
    method: str = "pc"
    significance_level: float = 0.05
    max_cond_vars: int = 3

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        allowed = {"pc", "lingam", "ges"}
        if v not in allowed:
            raise ValueError(f"method must be one of {allowed}")
        return v


class CausalModelConfig(BaseModel):
    estimator: str = "econml_dml"
    treatment: str = "reactor_temp"
    outcome: str = "product_yield"
    common_causes: list[str] = ["catalyst_type", "coolant_flow_rate", "ph_level"]
    effect_modifiers: list[str] = ["catalyst_type"]

    @field_validator("estimator")
    @classmethod
    def validate_estimator(cls, v: str) -> str:
        allowed = {"dowhy_linear", "econml_dml", "econml_causal_forest"}
        if v not in allowed:
            raise ValueError(f"estimator must be one of {allowed}")
        return v


class DetectionConfig(BaseModel):
    isolation_forest_contamination: float = 0.05
    cusum_threshold: float = 5.0
    cusum_drift: float = 0.5
    window_size: int = 20


class ControlConfig(BaseModel):
    target_product_yield: float = 87.0
    target_tolerance: float = 1.0
    max_temp_adjustment: float = 5.0
    max_cooling_adjustment: float = 10.0


class ServingConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class AppConfig(BaseModel):
    simulation: SimulationConfig = SimulationConfig()
    causal_graph: CausalGraphConfig = CausalGraphConfig()
    causal_model: CausalModelConfig = CausalModelConfig()
    detection: DetectionConfig = DetectionConfig()
    control: ControlConfig = ControlConfig()
    serving: ServingConfig = ServingConfig()


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_config(path: str = "config/config.yaml") -> AppConfig:
    """Load and validate configuration from YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        logger.warning(f"Config file {path} not found, using defaults")
        return AppConfig()
    with open(config_path) as f:
        data: dict[str, Any] = yaml.safe_load(f)
    config = AppConfig(**data)
    logger.info(f"Loaded config from {path}")
    return config
