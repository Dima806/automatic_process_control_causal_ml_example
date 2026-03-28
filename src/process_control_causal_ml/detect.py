"""Anomaly / deviation detection: Isolation Forest + CUSUM."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from process_control_causal_ml.utils import DetectionConfig, load_config, logger

DATA_DIR = Path("data")

# Continuous process variables used for detection (exclude metadata and categoricals)
FEATURE_COLS = [
    "coolant_flow_rate",
    "reactor_temp",
    "pressure",
    "ph_level",
    "reaction_rate",
    "product_yield",
]


@dataclass
class AnomalyResult:
    flag: bool
    score: float  # isolation forest anomaly score (lower = more anomalous)
    type: str  # "isolation_forest" | "cusum" | "none"
    variable: str  # most anomalous variable or "multivariate"


@dataclass
class CusumState:
    """Per-variable CUSUM running state."""

    mean: float = 0.0
    std: float = 1.0
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0


@dataclass
class AnomalyDetector:
    iso_forest: IsolationForest
    scaler: StandardScaler
    iso_threshold: float  # score below which a point is anomalous
    cusum_states: dict[str, CusumState]
    feature_cols: list[str]
    config: DetectionConfig


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------


def run_cusum(series: pd.Series, config: DetectionConfig) -> pd.Series:
    """Compute per-observation CUSUM statistic for a univariate series.

    Returns a Series of CUSUM values; values above config.cusum_threshold
    indicate a detected shift.
    """
    mu = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)

    k = config.cusum_drift

    normalized = (series.values - mu) / std
    cusum_pos = np.zeros(len(series))
    cusum_neg = np.zeros(len(series))

    for i in range(1, len(series)):
        cusum_pos[i] = max(0.0, cusum_pos[i - 1] + normalized[i] - k)
        cusum_neg[i] = max(0.0, cusum_neg[i - 1] - normalized[i] - k)

    combined = np.maximum(cusum_pos, cusum_neg)
    return pd.Series(combined, index=series.index, name=f"cusum_{series.name}")


def _update_cusum_state(
    state: CusumState, value: float, config: DetectionConfig
) -> tuple[float, bool]:
    """Update a single-variable CUSUM state with a new observation.

    Returns (cusum_value, alarm_flag).
    """
    k = config.cusum_drift
    h = config.cusum_threshold

    normalized = (value - state.mean) / max(state.std, 1e-8)
    state.cusum_pos = max(0.0, state.cusum_pos + normalized - k)
    state.cusum_neg = max(0.0, state.cusum_neg - normalized - k)

    cusum_val = max(state.cusum_pos, state.cusum_neg)
    alarm = cusum_val > h
    return cusum_val, alarm


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_detector(df: pd.DataFrame, config: DetectionConfig) -> AnomalyDetector:
    """Train Isolation Forest and compute CUSUM baseline statistics on normal data."""
    normal = df[~df["anomaly_flag"]].copy() if "anomaly_flag" in df.columns else df.copy()

    # Use only available feature columns
    available_cols = [c for c in FEATURE_COLS if c in normal.columns]

    X_normal = normal[available_cols].values.astype(float)

    # Fit scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_normal)

    # Fit Isolation Forest
    iso = IsolationForest(
        contamination=config.isolation_forest_contamination,
        random_state=42,
        n_estimators=100,
    )
    iso.fit(X_scaled)

    # Determine threshold from training data
    train_scores = iso.score_samples(X_scaled)
    iso_threshold = float(np.percentile(train_scores, config.isolation_forest_contamination * 100))

    # Compute per-variable CUSUM baseline (mean, std)
    cusum_states: dict[str, CusumState] = {}
    for col in available_cols:
        series = normal[col].astype(float)
        cusum_states[col] = CusumState(
            mean=float(series.mean()),
            std=float(series.std()),
            cusum_pos=0.0,
            cusum_neg=0.0,
        )

    logger.info(f"Detector trained on {len(normal)} normal samples, {len(available_cols)} features")
    logger.info(f"Isolation Forest threshold: {iso_threshold:.4f}")

    return AnomalyDetector(
        iso_forest=iso,
        scaler=scaler,
        iso_threshold=iso_threshold,
        cusum_states=cusum_states,
        feature_cols=available_cols,
        config=config,
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def detect_anomaly(reading: dict[str, float], detector: AnomalyDetector) -> AnomalyResult:
    """Detect anomaly in a single sensor reading using two-layer detection.

    Layer 1: Isolation Forest (multivariate global score)
    Layer 2: CUSUM per variable (sequential drift detection)

    Returns AnomalyResult with flag, score, type, and most anomalous variable.
    """
    # Build feature vector
    feature_vals = np.array(
        [reading.get(col, 0.0) for col in detector.feature_cols], dtype=float
    ).reshape(1, -1)

    # Scale
    feature_scaled = detector.scaler.transform(feature_vals)

    # Layer 1: Isolation Forest
    iso_score = float(detector.iso_forest.score_samples(feature_scaled)[0])
    iso_anomaly = iso_score < detector.iso_threshold

    # Layer 2: CUSUM per variable
    cusum_alarms: dict[str, float] = {}
    for col in detector.feature_cols:
        if col in reading:
            state = detector.cusum_states[col]
            cusum_val, alarm = _update_cusum_state(state, float(reading[col]), detector.config)
            if alarm:
                cusum_alarms[col] = cusum_val

    cusum_anomaly = len(cusum_alarms) > 0

    if iso_anomaly or cusum_anomaly:
        if cusum_anomaly:
            # Identify the most anomalous variable (highest CUSUM value)
            most_anomalous = max(cusum_alarms, key=cusum_alarms.__getitem__)
            return AnomalyResult(
                flag=True,
                score=iso_score,
                type="cusum",
                variable=most_anomalous,
            )
        else:
            return AnomalyResult(
                flag=True,
                score=iso_score,
                type="isolation_forest",
                variable="multivariate",
            )

    return AnomalyResult(flag=False, score=iso_score, type="none", variable="none")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(train: bool = False) -> None:
    config = load_config()
    DATA_DIR.mkdir(exist_ok=True)

    data_path = DATA_DIR / "process_data.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Run 'make simulate' first.")

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded data: {df.shape}")

    if train:
        detector = train_detector(df, config.detection)
        joblib.dump(detector, DATA_DIR / "detector.pkl")
        logger.info(f"Detector saved to {DATA_DIR / 'detector.pkl'}")

        # Evaluate on full dataset
        _evaluate_detector(df, detector)


def _evaluate_detector(df: pd.DataFrame, detector: AnomalyDetector) -> None:
    """Evaluate the trained detector on the full dataset and log F1 score."""
    from sklearn.metrics import classification_report, f1_score

    available_cols = [c for c in detector.feature_cols if c in df.columns]
    X = df[available_cols].values.astype(float)
    X_scaled = detector.scaler.transform(X)
    scores = detector.iso_forest.score_samples(X_scaled)
    preds = (scores < detector.iso_threshold).astype(int)

    if "anomaly_flag" in df.columns:
        y_true = df["anomaly_flag"].astype(int).values
        f1 = f1_score(y_true, preds, zero_division=0)
        logger.info(f"Isolation Forest F1: {f1:.4f}")
        logger.info(f"\n{classification_report(y_true, preds, target_names=['normal', 'anomaly'])}")


if __name__ == "__main__":
    # Fix pickle serialisation when run via `python -m`.
    # Python sets __name__ = '__main__', so dataclasses get __module__ = '__main__'.
    # joblib/pickle then stores them as '__main__.ClassName', which fails to
    # deserialise in other processes (e.g. uvicorn --reload uses '__mp_main__').
    # Fix: (1) register this module under its real dotted name in sys.modules,
    # then (2) patch __module__ on each dataclass so pickle uses that name.
    # Both steps are required: the registration makes the identity check pass,
    # the __module__ patch ensures pickle uses the stable dotted path.
    import sys as _sys

    if __spec__ is not None:
        _real_name = __spec__.name
        _sys.modules.setdefault(_real_name, _sys.modules[__name__])
        for _cls in (AnomalyResult, CusumState, AnomalyDetector):
            _cls.__module__ = _real_name

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    main(train=args.train)
