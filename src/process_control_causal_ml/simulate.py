"""Structural Causal Model (SCM) data generation for the chemical process simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from process_control_causal_ml.utils import SimulationConfig, load_config, logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "batch_id",
    "timestamp",
    "catalyst_type",
    "coolant_flow_rate",
    "reactor_temp",
    "pressure",
    "ph_level",
    "reaction_rate",
    "product_yield",
    "anomaly_flag",
    "anomaly_type",
]

DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# SCM generation
# ---------------------------------------------------------------------------


def generate_process_data(config: SimulationConfig) -> pd.DataFrame:
    """Generate synthetic process data using the Structural Causal Model.

    Returns a DataFrame with all process variables plus batch_id, timestamp,
    anomaly_flag (False), and anomaly_type ('none').
    """
    rng = np.random.default_rng(config.random_seed)
    n = config.n_batches

    # Exogenous variables
    catalyst_type = rng.choice(config.catalyst_types, size=n, p=[0.4, 0.35, 0.25])
    coolant_flow_rate = rng.normal(loc=50.0, scale=5.0, size=n)

    # Catalyst temperature effects
    catalyst_effect = np.array([config.catalyst_temp_effects[ct] for ct in catalyst_type])

    # reactor_temp = 180 + catalyst_effect - 0.20 * coolant_flow_rate + N(0, 1)
    reactor_temp = 180.0 + catalyst_effect - 0.20 * coolant_flow_rate + rng.normal(0.0, 1.0, size=n)

    # pressure = 2.5 + 0.06 * reactor_temp + N(0, 0.15)
    pressure = 2.5 + 0.06 * reactor_temp + rng.normal(0.0, 0.15, size=n)

    # ph_level = 7.0 - 0.05 * pressure - 0.02 * coolant_flow_rate + N(0, 0.06)
    ph_level = 7.0 - 0.05 * pressure - 0.02 * coolant_flow_rate + rng.normal(0.0, 0.06, size=n)

    # reaction_rate = 12 + 0.35 * reactor_temp + interaction(rt, catalyst) - 0.6 * ph_level + N(0, 0.6)
    # Interaction: catalyst modulates temperature coefficient
    # A: +0.0 * rt, B: +0.05 * rt, C: -0.03 * rt
    catalyst_interaction_coeff = np.array(
        [{"A": 0.0, "B": 0.05, "C": -0.03}[ct] for ct in catalyst_type]
    )
    reaction_rate = (
        12.0
        + 0.35 * reactor_temp
        + catalyst_interaction_coeff * reactor_temp
        - 0.6 * ph_level
        + rng.normal(0.0, 0.6, size=n)
    )

    # product_yield = 85 + 0.30 * reaction_rate + 0.08 * reactor_temp - 0.40 * pressure + N(0, 0.25)
    product_yield = (
        85.0
        + 0.30 * reaction_rate
        + 0.08 * reactor_temp
        - 0.40 * pressure
        + rng.normal(0.0, 0.25, size=n)
    )

    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")

    df = pd.DataFrame(
        {
            "batch_id": np.arange(n, dtype=int),
            "timestamp": timestamps,
            "catalyst_type": catalyst_type,
            "coolant_flow_rate": coolant_flow_rate,
            "reactor_temp": reactor_temp,
            "pressure": pressure,
            "ph_level": ph_level,
            "reaction_rate": reaction_rate,
            "product_yield": product_yield,
            "anomaly_flag": False,
            "anomaly_type": "none",
        }
    )

    return df


def _inject_drift(
    df: pd.DataFrame, rng: np.random.Generator, n: int, n_windows: int, used: set[int]
) -> None:
    """Inject gradual reactor_temp drift over 50-batch windows (in-place)."""
    for _ in range(n_windows):
        start = int(rng.integers(0, n - 60))
        for j, idx in enumerate(range(start, min(start + 50, n))):
            if idx not in used:
                df.at[idx, "reactor_temp"] = df.at[idx, "reactor_temp"] + (j / 50) * 10.0
                df.at[idx, "anomaly_flag"] = True
                df.at[idx, "anomaly_type"] = "drift"
                used.add(idx)


def _inject_step_change(
    df: pd.DataFrame, rng: np.random.Generator, n: int, budget: int, used: set[int]
) -> None:
    """Inject +3σ pressure step changes (in-place)."""
    pressure_std = float(df["pressure"].std())
    available = [i for i in range(n) if i not in used]
    indices = rng.choice(available, size=min(budget, len(available)), replace=False)
    for idx in indices:
        df.at[idx, "pressure"] = df.at[idx, "pressure"] + 3 * pressure_std
        df.at[idx, "anomaly_flag"] = True
        df.at[idx, "anomaly_type"] = "step_change"
        used.add(idx)


def _inject_sensor_noise(
    df: pd.DataFrame, rng: np.random.Generator, n: int, budget: int, used: set[int]
) -> None:
    """Inject ×5 variance sensor noise on ph_level (in-place)."""
    ph_std = float(df["ph_level"].std())
    available = [i for i in range(n) if i not in used]
    indices = rng.choice(available, size=min(budget, len(available)), replace=False)
    for idx in indices:
        df.at[idx, "ph_level"] = df.at[idx, "ph_level"] + rng.normal(0.0, ph_std * 5)
        df.at[idx, "anomaly_flag"] = True
        df.at[idx, "anomaly_type"] = "sensor_noise"
        used.add(idx)


def inject_anomalies(df: pd.DataFrame, config: SimulationConfig) -> pd.DataFrame:
    """Inject three types of anomalies into ~5% of batches.

    Types:
    - drift: gradual mean shift of +10°C over 50 consecutive batches on reactor_temp
    - step_change: sudden +3σ jump on pressure
    - sensor_noise: variance spike ×5 on ph_level
    """
    rng = np.random.default_rng(config.random_seed + 1)
    df = df.copy()
    n = len(df)
    anomaly_budget = int(n * config.anomaly_fraction) // 3
    n_drift_windows = max(1, anomaly_budget // 50)
    used_indices: set[int] = set()

    _inject_drift(df, rng, n, n_drift_windows, used_indices)
    _inject_step_change(df, rng, n, anomaly_budget, used_indices)
    _inject_sensor_noise(df, rng, n, anomaly_budget, used_indices)

    actual_frac = df["anomaly_flag"].mean()
    logger.info(
        f"Injected anomalies: {df['anomaly_flag'].sum()} batches ({actual_frac:.3f} fraction)"
    )
    return df


def _check_column_ranges(df: pd.DataFrame) -> None:
    """Raise ValueError if more than 10% of values fall outside expected SCM ranges."""
    checks: list[tuple[str, float, float]] = [
        ("reactor_temp", 140.0, 230.0),
        ("pressure", 9.0, 17.0),
        ("ph_level", 3.0, 8.0),
        ("product_yield", 90.0, 140.0),
        ("coolant_flow_rate", 20.0, 80.0),
    ]
    for col, lo, hi in checks:
        out = ~df[col].between(lo, hi)
        if out.mean() > 0.10:
            raise ValueError(f"{col}: {out.sum()} values out of range [{lo}, {hi}]")


def validate_data(df: pd.DataFrame) -> None:
    """Validate process data against expected ranges and schema.

    Raises ValueError if any validation check fails.
    """
    logger.info("Validating process data schema and ranges...")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    null_counts = df[REQUIRED_COLUMNS].isnull().sum()
    if null_counts.any():
        raise ValueError(f"Null values found:\n{null_counts[null_counts > 0]}")

    _check_column_ranges(df)

    bad_cats = set(df["catalyst_type"].unique()) - {"A", "B", "C"}
    if bad_cats:
        raise ValueError(f"Unexpected catalyst_type values: {bad_cats}")

    frac = df["anomaly_flag"].mean()
    if not (0.02 <= frac <= 0.12):
        raise ValueError(f"Anomaly fraction {frac:.3f} outside expected range [0.02, 0.12]")

    logger.info(f"Validation passed. Shape: {df.shape}, anomaly_fraction: {frac:.3f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(validate_only: bool = False) -> None:
    config = load_config()
    DATA_DIR.mkdir(exist_ok=True)

    out_path = DATA_DIR / "process_data.parquet"

    if validate_only:
        if not out_path.exists():
            raise FileNotFoundError(f"{out_path} does not exist. Run 'make simulate' first.")
        df = pd.read_parquet(out_path)
        validate_data(df)
        return

    logger.info(f"Generating {config.simulation.n_batches} batches...")
    df = generate_process_data(config.simulation)
    df = inject_anomalies(df, config.simulation)
    validate_data(df)

    df.to_parquet(out_path, index=False)
    logger.info(f"Saved {len(df)} rows to {out_path}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Anomaly types:\n{df['anomaly_type'].value_counts()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    main(validate_only=args.validate_only)
