"""Tests for control.py."""

from __future__ import annotations

import pytest

from process_control_causal_ml.control import (
    VARIABLE_TO_CONTROL_INPUT,
    CorrectiveAction,
    recommend_action,
)
from process_control_causal_ml.detect import AnomalyResult


@pytest.fixture
def normal_anomaly() -> AnomalyResult:
    return AnomalyResult(flag=False, score=-0.05, type="none", variable="none")


@pytest.fixture
def reactor_temp_anomaly() -> AnomalyResult:
    return AnomalyResult(flag=True, score=-0.25, type="cusum", variable="reactor_temp")


@pytest.fixture
def pressure_anomaly() -> AnomalyResult:
    return AnomalyResult(flag=True, score=-0.30, type="isolation_forest", variable="pressure")


@pytest.fixture
def low_yield_state() -> dict:
    return {
        "catalyst_type": "B",
        "coolant_flow_rate": 50.0,
        "reactor_temp": 181.0,
        "pressure": 2.7,
        "ph_level": 6.85,
        "reaction_rate": 72.0,
        "product_yield": 83.0,
    }


@pytest.fixture
def normal_state() -> dict:
    return {
        "catalyst_type": "A",
        "coolant_flow_rate": 50.0,
        "reactor_temp": 183.0,
        "pressure": 2.75,
        "ph_level": 6.9,
        "reaction_rate": 74.0,
        "product_yield": 86.5,
    }


def test_no_op_when_no_anomaly(
    normal_anomaly, trained_causal_model, normal_state, control_config
) -> None:
    action = recommend_action(normal_anomaly, trained_causal_model, normal_state, control_config)
    assert isinstance(action, CorrectiveAction)
    assert action.delta == 0.0
    assert action.variable == "none"


def test_action_for_anomaly(
    reactor_temp_anomaly, trained_causal_model, low_yield_state, control_config
) -> None:
    action = recommend_action(
        reactor_temp_anomaly, trained_causal_model, low_yield_state, control_config
    )
    assert isinstance(action, CorrectiveAction)
    assert action.variable != "none"


def test_temp_delta_within_bounds(
    reactor_temp_anomaly, trained_causal_model, low_yield_state, control_config
) -> None:
    action = recommend_action(
        reactor_temp_anomaly, trained_causal_model, low_yield_state, control_config
    )
    if action.variable == "reactor_temp":
        assert abs(action.delta) <= control_config.max_temp_adjustment


def test_cooling_delta_within_bounds(
    pressure_anomaly, trained_causal_model, low_yield_state, control_config
) -> None:
    action = recommend_action(
        pressure_anomaly, trained_causal_model, low_yield_state, control_config
    )
    if action.variable == "coolant_flow_rate":
        assert abs(action.delta) <= control_config.max_cooling_adjustment


def test_action_fields_types(
    reactor_temp_anomaly, trained_causal_model, low_yield_state, control_config
) -> None:
    action = recommend_action(
        reactor_temp_anomaly, trained_causal_model, low_yield_state, control_config
    )
    assert isinstance(action.variable, str)
    assert isinstance(action.current, float)
    assert isinstance(action.recommended, float)
    assert isinstance(action.delta, float)
    assert 0.0 <= action.confidence <= 1.0
    assert len(action.reasoning) > 0


def test_recommended_equals_current_plus_delta(
    reactor_temp_anomaly, trained_causal_model, low_yield_state, control_config
) -> None:
    action = recommend_action(
        reactor_temp_anomaly, trained_causal_model, low_yield_state, control_config
    )
    assert abs(action.recommended - (action.current + action.delta)) < 1e-6


def test_variable_mapping_coverage() -> None:
    expected = {
        "reactor_temp",
        "pressure",
        "ph_level",
        "reaction_rate",
        "product_yield",
        "multivariate",
        "none",
    }
    assert expected.issubset(set(VARIABLE_TO_CONTROL_INPUT.keys()))


def test_all_anomaly_variables_produce_action(
    trained_causal_model, low_yield_state, control_config
) -> None:
    """Every mapped anomaly variable should produce a CorrectiveAction."""
    for var in VARIABLE_TO_CONTROL_INPUT:
        anomaly = AnomalyResult(flag=True, score=-0.3, type="cusum", variable=var)
        action = recommend_action(anomaly, trained_causal_model, low_yield_state, control_config)
        assert isinstance(action, CorrectiveAction)
        assert action.variable != "none" or var == "none"


def test_action_above_target_yield_negative_or_zero_delta(
    trained_causal_model, control_config
) -> None:
    """When yield is already above target, corrective delta should be <= 0."""
    high_yield_state = {
        "catalyst_type": "A",
        "coolant_flow_rate": 50.0,
        "reactor_temp": 185.0,
        "pressure": 2.8,
        "ph_level": 6.9,
        "reaction_rate": 76.0,
        "product_yield": 92.0,  # above target 87
    }
    anomaly = AnomalyResult(flag=True, score=-0.2, type="cusum", variable="reactor_temp")
    action = recommend_action(anomaly, trained_causal_model, high_yield_state, control_config)
    if action.variable == "reactor_temp":
        assert action.delta <= 0


def test_fallback_model_dowhy_value_path(low_yield_state, control_config) -> None:
    """recommend_action should handle a DoWhy-style model with a .value attribute."""

    class MockDoWhyEstimate:
        value = 0.15

    anomaly = AnomalyResult(flag=True, score=-0.2, type="cusum", variable="reactor_temp")
    action = recommend_action(anomaly, MockDoWhyEstimate(), low_yield_state, control_config)
    assert isinstance(action, CorrectiveAction)


def test_zero_cate_returns_no_delta(low_yield_state, control_config) -> None:
    """When CATE is effectively zero, recommend_action must return delta=0."""
    import numpy as np

    class ZeroCateModel:
        def effect(self, X):
            return np.zeros((1, 1))

        def effect_interval(self, X, alpha=0.05):
            return np.zeros((1, 1)), np.zeros((1, 1))

    anomaly = AnomalyResult(flag=True, score=-0.3, type="cusum", variable="reactor_temp")
    action = recommend_action(anomaly, ZeroCateModel(), low_yield_state, control_config)
    assert action.delta == 0.0


def test_ph_level_anomaly_controls_coolant(
    trained_causal_model, low_yield_state, control_config
) -> None:
    """ph_level anomaly should map to coolant_flow_rate control input."""
    anomaly = AnomalyResult(flag=True, score=-0.3, type="cusum", variable="ph_level")
    action = recommend_action(anomaly, trained_causal_model, low_yield_state, control_config)
    assert action.variable == "coolant_flow_rate"
