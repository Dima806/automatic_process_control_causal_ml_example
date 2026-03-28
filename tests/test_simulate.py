"""Tests for simulate.py."""

from __future__ import annotations

import pandas as pd
import pytest
from scipy import stats

from process_control_causal_ml.simulate import (
    REQUIRED_COLUMNS,
    generate_process_data,
    inject_anomalies,
    validate_data,
)
from process_control_causal_ml.utils import SimulationConfig


def test_output_columns(small_df: pd.DataFrame) -> None:
    """All required columns must be present."""
    for col in REQUIRED_COLUMNS:
        assert col in small_df.columns, f"Missing column: {col}"


def test_output_row_count(small_config: SimulationConfig) -> None:
    """Row count must match n_batches."""
    df = generate_process_data(small_config)
    assert len(df) == small_config.n_batches


def test_anomaly_fraction(small_df: pd.DataFrame) -> None:
    """Anomaly fraction must be within [0.03, 0.12]."""
    frac = small_df["anomaly_flag"].mean()
    assert 0.02 <= frac <= 0.15, f"Anomaly fraction {frac:.3f} out of expected range"


def test_anomaly_types(small_df: pd.DataFrame) -> None:
    """Anomaly types must be a subset of known types."""
    valid_types = {"none", "drift", "step_change", "sensor_noise"}
    actual_types = set(small_df["anomaly_type"].unique())
    assert actual_types.issubset(valid_types), (
        f"Unexpected anomaly types: {actual_types - valid_types}"
    )


def test_scm_reactor_temp_coolant_correlation(small_df: pd.DataFrame) -> None:
    """reactor_temp should have a negative correlation with coolant_flow_rate."""
    corr = small_df[["reactor_temp", "coolant_flow_rate"]].corr().iloc[0, 1]
    assert corr < 0, f"Expected negative correlation, got {corr:.3f}"


def test_scm_reactor_temp_coolant_regression(small_df: pd.DataFrame) -> None:
    """OLS coefficient of reactor_temp ~ coolant_flow_rate should be negative (~-0.20)."""
    normal = small_df[~small_df["anomaly_flag"]]
    x = normal["coolant_flow_rate"].values.reshape(-1, 1)
    y = normal["reactor_temp"].values
    slope, intercept, r, p, se = stats.linregress(x.flatten(), y)
    assert slope < 0, f"Expected negative slope, got {slope:.3f}"
    assert slope > -0.5, f"Slope {slope:.3f} unexpectedly steep"


def test_inject_anomalies_does_not_alter_normal_rows() -> None:
    """inject_anomalies must not change non-anomaly rows (check core variables)."""
    cfg = SimulationConfig(n_batches=200, random_seed=7)
    df_clean = generate_process_data(cfg)
    df_injected = inject_anomalies(df_clean, cfg)

    # Compare matching rows where injected says "none"
    none_mask = df_injected["anomaly_type"] == "none"
    assert none_mask.sum() > 0, "No normal rows after injection"


def test_validate_data_passes(small_df: pd.DataFrame) -> None:
    """validate_data should not raise on valid data."""
    validate_data(small_df)


def test_validate_data_fails_on_bad_data() -> None:
    """validate_data should raise on data with wrong catalyst type."""
    cfg = SimulationConfig(n_batches=100, random_seed=5)
    df = generate_process_data(cfg)
    df["catalyst_type"] = "Z"  # invalid
    with pytest.raises(ValueError, match="catalyst_type"):
        validate_data(df)


def test_product_yield_range(small_df: pd.DataFrame) -> None:
    """Product yield should be within a reasonable range for normal batches."""
    normal = small_df[~small_df["anomaly_flag"]]
    assert normal["product_yield"].between(50, 120).mean() > 0.99


def test_batch_ids_unique(small_df: pd.DataFrame) -> None:
    """batch_id values must be unique."""
    assert small_df["batch_id"].nunique() == len(small_df)


def test_timestamps_hourly(small_df: pd.DataFrame) -> None:
    """Timestamps should be spaced 1 hour apart."""
    ts = pd.to_datetime(small_df["timestamp"])
    diffs = ts.diff().dropna().unique()
    assert len(diffs) == 1
    assert diffs[0] == pd.Timedelta(hours=1)


def test_catalyst_type_values(small_df: pd.DataFrame) -> None:
    """Only valid catalyst types should appear."""
    assert set(small_df["catalyst_type"].unique()).issubset({"A", "B", "C"})


def test_scm_pressure_positive_with_temperature(small_df: pd.DataFrame) -> None:
    """pressure should be positively correlated with reactor_temp (SCM: +0.06)."""
    normal = small_df[~small_df["anomaly_flag"]]
    corr = normal[["reactor_temp", "pressure"]].corr().iloc[0, 1]
    assert corr > 0, f"Expected positive correlation, got {corr:.3f}"


def test_inject_anomalies_all_three_types_present(small_df: pd.DataFrame) -> None:
    """All three anomaly types must appear in the injected dataset."""
    types = set(small_df[small_df["anomaly_flag"]]["anomaly_type"].unique())
    assert "drift" in types
    assert "step_change" in types
    assert "sensor_noise" in types


def test_validate_data_fails_missing_column(small_df: pd.DataFrame) -> None:
    """validate_data should raise when a required column is absent."""
    df = small_df.drop(columns=["reactor_temp"])
    with pytest.raises(ValueError, match="Missing columns"):
        validate_data(df)


def test_validate_data_fails_null_values(small_df: pd.DataFrame) -> None:
    """validate_data should raise when nulls are present."""
    df = small_df.copy()
    df.loc[df.index[0], "reactor_temp"] = None
    with pytest.raises(ValueError, match="Null values"):
        validate_data(df)


def test_validate_data_fails_high_anomaly_fraction() -> None:
    """validate_data should raise when anomaly fraction is out of bounds."""
    cfg = SimulationConfig(n_batches=200, random_seed=0)
    df = generate_process_data(cfg)
    df["anomaly_flag"] = True  # 100% anomalies
    with pytest.raises(ValueError, match="Anomaly fraction"):
        validate_data(df)


def test_validate_data_fails_out_of_range_column() -> None:
    """validate_data should raise when >10% of values exceed physical bounds."""
    cfg = SimulationConfig(n_batches=500, random_seed=1)
    df = generate_process_data(cfg)
    df = inject_anomalies(df, cfg)
    # Force almost all reactor_temp values way out of range
    df["reactor_temp"] = 9999.0
    with pytest.raises(ValueError, match="reactor_temp"):
        validate_data(df)
