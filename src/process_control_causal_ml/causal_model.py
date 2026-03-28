"""Causal treatment effect estimation using DoWhy and EconML."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from process_control_causal_ml.utils import CausalModelConfig, load_config, logger

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_categoricals(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """One-hot encode string columns in frame[cols]; return encoded float DataFrame."""
    result = frame[cols].copy()
    cat_cols = [c for c in cols if pd.api.types.is_string_dtype(result[c])]
    if cat_cols:
        result = pd.get_dummies(result, columns=cat_cols, drop_first=False)
    return result.astype(float)


def _prepare_data(
    df: pd.DataFrame, config: CausalModelConfig
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Prepare Y, T, X (effect modifiers), W (confounders) for EconML.

    Categorical columns are one-hot encoded.
    Returns: Y (outcome), T (treatment), X (effect modifiers), W (confounders).
    """
    normal = df[~df["anomaly_flag"]].copy() if "anomaly_flag" in df.columns else df.copy()

    y = normal[config.outcome].astype(float)
    t = normal[config.treatment].astype(float)

    x_df = _encode_categoricals(normal, config.effect_modifiers)

    w_cols = [c for c in config.common_causes if c != config.treatment and c != config.outcome]
    w_df = _encode_categoricals(normal, w_cols)

    return y, t, x_df, w_df


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def _fit_dowhy_linear(df: pd.DataFrame, config: CausalModelConfig) -> tuple[Any, Any]:
    """Fit DoWhy linear regression estimator."""
    from dowhy import CausalModel

    normal = df[~df["anomaly_flag"]].copy() if "anomaly_flag" in df.columns else df.copy()

    # Prepare data: one-hot encode categoricals for DoWhy
    data_dummies = normal.copy()
    if "catalyst_type" in data_dummies.columns:
        data_dummies = pd.get_dummies(data_dummies, columns=["catalyst_type"], drop_first=False)

    # Update config common_causes to use dummy names
    common_causes = []
    for c in config.common_causes:
        if c == "catalyst_type":
            common_causes.extend(
                [col for col in data_dummies.columns if col.startswith("catalyst_type_")]
            )
        else:
            common_causes.append(c)

    model = CausalModel(
        data=data_dummies,
        treatment=config.treatment,
        outcome=config.outcome,
        common_causes=common_causes,
    )
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified,
        method_name="backdoor.linear_regression",
        confidence_intervals=True,
    )
    logger.info(f"DoWhy linear ATE: {estimate.value:.4f}")
    return model, estimate


def _fit_econml_dml(df: pd.DataFrame, config: CausalModelConfig) -> Any:
    """Fit EconML LinearDML estimator."""
    from econml.dml import LinearDML
    from sklearn.linear_model import LassoCV, RidgeCV

    y, t, x_df, w_df = _prepare_data(df, config)

    est = LinearDML(
        model_y=LassoCV(cv=3, max_iter=2000),
        model_t=RidgeCV(cv=3),
        random_state=42,
        cv=3,
    )
    W = w_df.values if len(w_df.columns) > 0 else None
    X = x_df.values if len(x_df.columns) > 0 else None

    est.fit(Y=y.values, T=t.values, X=X, W=W)

    ate = float(np.mean(est.effect(X=X)))
    logger.info(f"EconML LinearDML ATE: {ate:.4f}")
    return est


def _fit_econml_causal_forest(df: pd.DataFrame, config: CausalModelConfig) -> Any:
    """Fit EconML CausalForestDML estimator."""
    from econml.dml import CausalForestDML
    from sklearn.linear_model import LassoCV, RidgeCV

    y, t, x_df, w_df = _prepare_data(df, config)

    est = CausalForestDML(
        model_y=LassoCV(cv=3, max_iter=2000),
        model_t=RidgeCV(cv=3),
        random_state=42,
        n_estimators=100,
        cv=3,
        discrete_treatment=False,
    )
    W = w_df.values if len(w_df.columns) > 0 else None
    X = x_df.values if len(x_df.columns) > 0 else None

    est.fit(Y=y.values, T=t.values, X=X, W=W)

    ate = float(np.mean(est.effect(X=X)))
    logger.info(f"EconML CausalForestDML ATE: {ate:.4f}")
    return est


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_causal_model(df: pd.DataFrame, config: CausalModelConfig) -> Any:
    """Train the causal model specified in config and return the fitted estimator."""
    logger.info(f"Training causal model: {config.estimator}")
    if config.estimator == "dowhy_linear":
        _, estimate = _fit_dowhy_linear(df, config)
        return estimate
    elif config.estimator == "econml_dml":
        return _fit_econml_dml(df, config)
    elif config.estimator == "econml_causal_forest":
        return _fit_econml_causal_forest(df, config)
    else:
        raise ValueError(f"Unknown estimator: {config.estimator}")


def estimate_ate(df: pd.DataFrame, config: CausalModelConfig) -> float:
    """Estimate the Average Treatment Effect of treatment on outcome."""
    y, t, x_df, w_df = _prepare_data(df, config)
    X = x_df.values if len(x_df.columns) > 0 else None

    if config.estimator == "dowhy_linear":
        _, estimate = _fit_dowhy_linear(df, config)
        return float(estimate.value)
    elif config.estimator in ("econml_dml", "econml_causal_forest"):
        est = train_causal_model(df, config)
        ate = float(np.mean(est.effect(X=X)))
        return ate
    else:
        raise ValueError(f"Unknown estimator: {config.estimator}")


def estimate_cate(df: pd.DataFrame, config: CausalModelConfig) -> pd.Series:
    """Estimate Conditional Average Treatment Effects (CATE) per observation."""
    normal = df[~df["anomaly_flag"]].copy() if "anomaly_flag" in df.columns else df.copy()
    y, t, x_df, w_df = _prepare_data(df, config)
    X = x_df.values if len(x_df.columns) > 0 else None

    # Use CausalForestDML for heterogeneous effects
    config_forest = config.model_copy(update={"estimator": "econml_causal_forest"})
    est = _fit_econml_causal_forest(df, config_forest)
    cate = est.effect(X=X)
    return pd.Series(cate.flatten(), name="cate", index=normal.index[: len(cate)])


def estimate_interaction_effects(
    df: pd.DataFrame, config: CausalModelConfig | None = None
) -> pd.DataFrame:
    """Estimate treatment effect heterogeneity by catalyst type (interaction effects).

    Returns a DataFrame with one row per catalyst type and columns:
    catalyst_type, mean_cate, std_cate, n_obs.
    """
    if config is None:
        config = load_config().causal_model

    normal = df[~df["anomaly_flag"]].copy() if "anomaly_flag" in df.columns else df.copy()
    y, t, x_df, w_df = _prepare_data(df, config)
    X = x_df.values if len(x_df.columns) > 0 else None

    est = _fit_econml_causal_forest(df, config)
    cate = est.effect(X=X).flatten()

    # Map back to catalyst types
    catalyst_col = normal["catalyst_type"].values[: len(cate)]
    rows = []
    for ct in sorted(set(catalyst_col)):
        mask = catalyst_col == ct
        rows.append(
            {
                "catalyst_type": ct,
                "mean_cate": float(np.mean(cate[mask])),
                "std_cate": float(np.std(cate[mask])),
                "n_obs": int(mask.sum()),
            }
        )
    result = pd.DataFrame(rows)
    logger.info(f"Interaction effects by catalyst type:\n{result.to_string()}")
    return result


def refute_estimate(model: Any, estimate: Any, n_simulations: int = 100) -> dict[str, Any]:
    """Run DoWhy refutation tests to validate the causal estimate."""
    results: dict[str, Any] = {}
    try:
        refuter1 = model.refute_estimate(
            model.identify_effect(proceed_when_unidentifiable=True),
            estimate,
            method_name="random_common_cause",
            num_simulations=n_simulations,
        )
        results["random_common_cause"] = {
            "new_effect": float(refuter1.new_effect),
            "refutation_result": str(refuter1.refutation_result),
        }
    except Exception as e:
        logger.warning(f"random_common_cause refutation failed: {e}")

    try:
        refuter2 = model.refute_estimate(
            model.identify_effect(proceed_when_unidentifiable=True),
            estimate,
            method_name="placebo_treatment_refuter",
            num_simulations=n_simulations,
        )
        results["placebo_treatment"] = {
            "new_effect": float(refuter2.new_effect),
            "refutation_result": str(refuter2.refutation_result),
        }
    except Exception as e:
        logger.warning(f"placebo_treatment refutation failed: {e}")

    try:
        refuter3 = model.refute_estimate(
            model.identify_effect(proceed_when_unidentifiable=True),
            estimate,
            method_name="data_subset_refuter",
            num_simulations=n_simulations,
        )
        results["data_subset"] = {
            "new_effect": float(refuter3.new_effect),
            "refutation_result": str(refuter3.refutation_result),
        }
    except Exception as e:
        logger.warning(f"data_subset refutation failed: {e}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(refute: bool = False) -> None:
    config = load_config()
    DATA_DIR.mkdir(exist_ok=True)

    data_path = DATA_DIR / "process_data.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Run 'make simulate' first.")

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded data: {df.shape}")

    # Train primary estimator
    model = train_causal_model(df, config.causal_model)

    # Save model
    joblib.dump({"model": model, "config": config.causal_model}, DATA_DIR / "causal_model.pkl")

    # Compute ATE
    y, t, x_df, w_df = _prepare_data(df, config.causal_model)
    X = x_df.values if len(x_df.columns) > 0 else None

    if config.causal_model.estimator in ("econml_dml", "econml_causal_forest"):
        ate = float(np.mean(model.effect(X=X)))
    else:
        ate = float(model.value) if hasattr(model, "value") else float("nan")

    # Interaction effects
    interaction_df = estimate_interaction_effects(df, config.causal_model)

    results: dict[str, Any] = {
        "estimator": config.causal_model.estimator,
        "treatment": config.causal_model.treatment,
        "outcome": config.causal_model.outcome,
        "ate": ate,
        "interaction_effects": interaction_df.to_dict(orient="records"),
    }

    if refute and config.causal_model.estimator == "dowhy_linear":
        logger.info("Running refutation tests...")
        dowhy_model, dowhy_estimate = _fit_dowhy_linear(df, config.causal_model)
        refutation = refute_estimate(dowhy_model, dowhy_estimate, n_simulations=50)
        results["refutation"] = refutation

    with open(DATA_DIR / "ate_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"ATE: {ate:.4f}")
    logger.info(f"Results saved to {DATA_DIR / 'ate_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refute", action="store_true")
    args = parser.parse_args()
    main(refute=args.refute)
