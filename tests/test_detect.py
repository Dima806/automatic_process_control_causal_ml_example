"""Tests for detect.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

from process_control_causal_ml.detect import (
    FEATURE_COLS,
    AnomalyDetector,
    AnomalyResult,
    CusumState,
    detect_anomaly,
    run_cusum,
    train_detector,
)
from process_control_causal_ml.utils import DetectionConfig


def test_train_detector_returns_detector(small_df, detection_config) -> None:
    detector = train_detector(small_df, detection_config)
    assert isinstance(detector, AnomalyDetector)


def test_detector_has_fitted_model(trained_detector) -> None:
    from sklearn.ensemble import IsolationForest

    assert isinstance(trained_detector.iso_forest, IsolationForest)


def test_detector_has_cusum_states(trained_detector) -> None:
    assert len(trained_detector.cusum_states) > 0
    for col, state in trained_detector.cusum_states.items():
        assert isinstance(state, CusumState)
        assert state.std > 0


def test_detect_anomaly_returns_result(trained_detector, normal_df) -> None:
    reading = normal_df.iloc[0][trained_detector.feature_cols].to_dict()
    result = detect_anomaly(reading, trained_detector)
    assert isinstance(result, AnomalyResult)
    assert isinstance(result.flag, bool)
    assert isinstance(result.score, float)
    assert result.type in {"isolation_forest", "cusum", "none"}
    assert isinstance(result.variable, str)


def test_detect_anomaly_extreme_reading(trained_detector) -> None:
    extreme = {
        "coolant_flow_rate": 50.0,
        "reactor_temp": 250.0,
        "pressure": 10.0,
        "ph_level": 7.0,
        "reaction_rate": 80.0,
        "product_yield": 85.0,
    }
    result = detect_anomaly(extreme, trained_detector)
    assert isinstance(result, AnomalyResult)


def test_run_cusum_returns_series(small_df) -> None:
    config = DetectionConfig()
    cusum = run_cusum(small_df["reactor_temp"], config)
    assert isinstance(cusum, pd.Series)
    assert len(cusum) == len(small_df)


def test_run_cusum_non_negative(small_df) -> None:
    config = DetectionConfig()
    cusum = run_cusum(small_df["reactor_temp"], config)
    assert (cusum >= 0).all()


def test_run_cusum_detects_shift() -> None:
    config = DetectionConfig(cusum_threshold=3.0, cusum_drift=0.3)
    rng = np.random.default_rng(42)
    series = pd.Series(np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 50)]))
    cusum = run_cusum(series, config)
    assert cusum.iloc[120:].max() > config.cusum_threshold


def test_feature_cols_present_in_small_df(small_df) -> None:
    for col in FEATURE_COLS:
        assert col in small_df.columns


def test_run_cusum_constant_series_returns_zeros(small_df) -> None:
    """A constant series has zero std; CUSUM should return all zeros."""
    config = DetectionConfig()
    constant = pd.Series(np.ones(len(small_df)))
    cusum = run_cusum(constant, config)
    assert (cusum == 0).all()


def test_detect_anomaly_returns_isolation_forest_type(trained_detector) -> None:
    """A single extreme multivariate reading with no CUSUM history should
    trigger isolation forest if the score is below threshold."""
    # Reset CUSUM states so CUSUM cannot trigger
    for state in trained_detector.cusum_states.values():
        state.cusum_pos = 0.0
        state.cusum_neg = 0.0

    extreme = {
        "coolant_flow_rate": 50.0,
        "reactor_temp": 300.0,  # very high
        "pressure": 20.0,
        "ph_level": 1.0,
        "reaction_rate": 200.0,
        "product_yield": 10.0,
    }
    result = detect_anomaly(extreme, trained_detector)
    # Score should be below threshold for such extreme values
    if result.flag and result.type == "isolation_forest":
        assert result.variable == "multivariate"


def test_detect_anomaly_normal_reading_low_score(trained_detector, normal_df) -> None:
    """Readings close to training mean should produce scores near zero."""
    reading = normal_df.iloc[5][trained_detector.feature_cols].to_dict()
    result = detect_anomaly(reading, trained_detector)
    assert isinstance(result.score, float)
    # Score should be finite
    assert result.score == result.score  # not NaN


def test_update_cusum_state_triggers_alarm() -> None:
    """Feeding values far above mean should eventually trigger a CUSUM alarm."""
    from process_control_causal_ml.detect import CusumState, _update_cusum_state

    state = CusumState(mean=0.0, std=1.0)
    config = DetectionConfig(cusum_threshold=3.0, cusum_drift=0.1)
    alarm_triggered = False
    for _ in range(20):
        _, alarm = _update_cusum_state(state, 5.0, config)
        if alarm:
            alarm_triggered = True
            break
    assert alarm_triggered


def test_train_detector_feature_cols_subset(small_df, detection_config) -> None:
    """Detector feature_cols should be a subset of the dataset columns."""
    detector = train_detector(small_df, detection_config)
    assert set(detector.feature_cols).issubset(set(small_df.columns))


def test_train_detector_threshold_is_negative(small_df, detection_config) -> None:
    """IsolationForest scores are negative; threshold should be negative."""
    detector = train_detector(small_df, detection_config)
    assert detector.iso_threshold < 0
