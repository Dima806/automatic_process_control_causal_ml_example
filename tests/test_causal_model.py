"""Tests for causal_model.py."""

from __future__ import annotations

import pandas as pd

from process_control_causal_ml.causal_model import (
    _encode_categoricals,
    _fit_econml_dml,
    _prepare_data,
    estimate_ate,
    estimate_interaction_effects,
    train_causal_model,
)


def test_prepare_data_shapes(small_df, causal_model_config) -> None:
    """_prepare_data should return Y, T, X, W with consistent row counts."""
    y, t, x_df, w_df = _prepare_data(small_df, causal_model_config)
    assert len(y) == len(t)
    assert len(y) == len(x_df)
    assert len(y) == len(w_df)


def test_prepare_data_no_nulls(small_df, causal_model_config) -> None:
    """Prepared arrays should have no null values."""
    y, t, x_df, w_df = _prepare_data(small_df, causal_model_config)
    assert not y.isnull().any()
    assert not t.isnull().any()
    assert not x_df.isnull().values.any()
    assert not w_df.isnull().values.any()


def test_fit_econml_dml_returns_model(small_df, causal_model_config) -> None:
    """_fit_econml_dml should return a fitted model object."""
    model = _fit_econml_dml(small_df, causal_model_config)
    assert model is not None
    assert hasattr(model, "effect")


def test_ate_is_float(small_df, causal_model_config) -> None:
    """estimate_ate should return a float."""
    ate = estimate_ate(small_df, causal_model_config)
    assert isinstance(ate, float)


def test_ate_sign_positive(small_df, causal_model_config) -> None:
    """ATE of reactor_temp on product_yield should be positive."""
    ate = estimate_ate(small_df, causal_model_config)
    assert ate > 0, f"Expected positive ATE, got {ate:.4f}"


def test_ate_reasonable_magnitude(small_df, causal_model_config) -> None:
    """ATE should be in a reasonable range (0.05 to 0.50 per °C)."""
    ate = estimate_ate(small_df, causal_model_config)
    assert 0.05 <= ate <= 0.5, f"ATE {ate:.4f} outside expected range [0.05, 0.50]"


def test_interaction_effects_shape(small_df, causal_model_config) -> None:
    """estimate_interaction_effects should return a DataFrame with 3 rows."""
    interaction_df = estimate_interaction_effects(small_df, causal_model_config)
    assert isinstance(interaction_df, pd.DataFrame)
    assert len(interaction_df) == 3, f"Expected 3 rows (A, B, C), got {len(interaction_df)}"


def test_interaction_effects_columns(small_df, causal_model_config) -> None:
    """Interaction effects DataFrame must have required columns."""
    interaction_df = estimate_interaction_effects(small_df, causal_model_config)
    required = {"catalyst_type", "mean_cate", "std_cate", "n_obs"}
    assert required.issubset(set(interaction_df.columns))


def test_interaction_effects_catalyst_types(small_df, causal_model_config) -> None:
    """Interaction effects should include all three catalyst types."""
    interaction_df = estimate_interaction_effects(small_df, causal_model_config)
    catalyst_types = set(interaction_df["catalyst_type"].values)
    assert catalyst_types == {"A", "B", "C"}


def test_encode_categoricals_one_hot(small_df) -> None:
    """_encode_categoricals should produce dummy columns for string cols."""
    result = _encode_categoricals(small_df, ["catalyst_type"])
    # drop_first=False: A, B, C all encoded (but we use drop_first=False in _encode_categoricals)
    assert result.dtypes.apply(lambda d: d == "float64" or str(d).startswith("float")).all()
    assert result.shape[0] == len(small_df)


def test_encode_categoricals_numeric_passthrough(small_df) -> None:
    """Numeric columns should pass through unchanged (just cast to float)."""
    result = _encode_categoricals(small_df, ["reactor_temp", "ph_level"])
    assert list(result.columns) == ["reactor_temp", "ph_level"]
    assert result.shape == (len(small_df), 2)


def test_prepare_data_excludes_anomaly_rows(small_df, causal_model_config) -> None:
    """Y, T, X, W should contain only normal (non-anomaly) rows."""
    y, t, x_df, w_df = _prepare_data(small_df, causal_model_config)
    n_normal = (~small_df["anomaly_flag"]).sum()
    assert len(y) == n_normal


def test_train_causal_model_returns_model_with_effect(small_df, causal_model_config) -> None:
    """train_causal_model with econml_dml should return a model with .effect()."""
    model = train_causal_model(small_df, causal_model_config)
    assert hasattr(model, "effect")


def test_train_causal_model_unknown_estimator_raises(small_df) -> None:
    from process_control_causal_ml.utils import CausalModelConfig

    cfg = CausalModelConfig.model_construct(
        estimator="unknown",
        treatment="reactor_temp",
        outcome="product_yield",
        common_causes=["coolant_flow_rate"],
        effect_modifiers=["catalyst_type"],
    )
    import pytest

    with pytest.raises(ValueError, match="Unknown estimator"):
        train_causal_model(small_df, cfg)
