"""Tests for utils.py — config models and loader."""

from __future__ import annotations

import pytest

from process_control_causal_ml.utils import (
    AppConfig,
    CausalGraphConfig,
    CausalModelConfig,
    SimulationConfig,
    load_config,
)


def test_load_config_returns_defaults_for_missing_file() -> None:
    """load_config should return defaults when the file does not exist."""
    cfg = load_config("nonexistent/path/config.yaml")
    assert isinstance(cfg, AppConfig)
    assert cfg.simulation.n_batches == 50000


def test_load_config_from_yaml(tmp_path) -> None:
    """load_config should parse a minimal YAML correctly."""
    yaml_content = "simulation:\n  n_batches: 123\n"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)
    cfg = load_config(str(config_file))
    assert cfg.simulation.n_batches == 123


def test_causal_graph_config_valid_methods() -> None:
    for method in ("pc", "lingam", "ges"):
        cfg = CausalGraphConfig(method=method)
        assert cfg.method == method


def test_causal_graph_config_rejects_invalid_method() -> None:
    with pytest.raises(ValueError, match="method"):
        CausalGraphConfig(method="invalid_algo")


def test_causal_model_config_rejects_invalid_estimator() -> None:
    with pytest.raises(ValueError, match="estimator"):
        CausalModelConfig(estimator="bad_estimator")


def test_simulation_config_defaults() -> None:
    cfg = SimulationConfig()
    assert cfg.random_seed == 42
    assert set(cfg.catalyst_types) == {"A", "B", "C"}
    assert cfg.anomaly_fraction == 0.05


def test_app_config_nests_all_sections() -> None:
    cfg = AppConfig()
    assert hasattr(cfg, "simulation")
    assert hasattr(cfg, "causal_graph")
    assert hasattr(cfg, "causal_model")
    assert hasattr(cfg, "detection")
    assert hasattr(cfg, "control")
    assert hasattr(cfg, "serving")
