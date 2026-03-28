"""Pytest fixtures shared across all test modules."""

from __future__ import annotations

import pandas as pd
import pytest

from process_control_causal_ml.simulate import generate_process_data, inject_anomalies
from process_control_causal_ml.utils import (
    AppConfig,
    CausalModelConfig,
    ControlConfig,
    DetectionConfig,
    SimulationConfig,
    load_config,
)


@pytest.fixture(scope="session")
def config() -> AppConfig:
    """Load default config (or use defaults if config.yaml not found)."""
    try:
        return load_config("config/config.yaml")
    except Exception:
        return AppConfig()


@pytest.fixture(scope="session")
def small_config() -> SimulationConfig:
    """Small simulation config for fast tests."""
    return SimulationConfig(n_batches=1000, random_seed=0)


@pytest.fixture(scope="session")
def small_df(small_config: SimulationConfig) -> pd.DataFrame:
    """Generate a small dataset for fast tests (1000 rows)."""
    df = generate_process_data(small_config)
    df = inject_anomalies(df, small_config)
    return df


@pytest.fixture(scope="session")
def normal_df(small_df: pd.DataFrame) -> pd.DataFrame:
    """Normal (non-anomaly) subset of the small dataset."""
    return small_df[~small_df["anomaly_flag"]].copy()


@pytest.fixture(scope="session")
def detection_config() -> DetectionConfig:
    return DetectionConfig(
        isolation_forest_contamination=0.05,
        cusum_threshold=5.0,
        cusum_drift=0.5,
        window_size=20,
    )


@pytest.fixture(scope="session")
def control_config() -> ControlConfig:
    return ControlConfig(
        target_product_yield=87.0,
        target_tolerance=1.0,
        max_temp_adjustment=5.0,
        max_cooling_adjustment=10.0,
    )


@pytest.fixture(scope="session")
def causal_model_config() -> CausalModelConfig:
    return CausalModelConfig(
        estimator="econml_dml",
        treatment="reactor_temp",
        outcome="product_yield",
        common_causes=["catalyst_type", "coolant_flow_rate", "ph_level"],
        effect_modifiers=["catalyst_type"],
    )


@pytest.fixture(scope="session")
def trained_detector(small_df: pd.DataFrame, detection_config: DetectionConfig):
    """Pre-trained AnomalyDetector on the small dataset."""
    from process_control_causal_ml.detect import train_detector

    return train_detector(small_df, detection_config)


@pytest.fixture(scope="session")
def trained_causal_model(small_df: pd.DataFrame, causal_model_config: CausalModelConfig):
    """Pre-trained EconML DML model on the small dataset."""
    from process_control_causal_ml.causal_model import _fit_econml_dml

    return _fit_econml_dml(small_df, causal_model_config)
