---
description: Add a new causal effect estimator to causal_model.py
---

To add a new estimator to this project:

1. **Add the estimator key** to `CausalModelConfig.validate_estimator` in `src/process_control_causal_ml/utils.py` — the allowed set is `{"dowhy_linear", "econml_dml", "econml_causal_forest"}`.

2. **Implement a `_fit_<name>` function** in `src/process_control_causal_ml/causal_model.py` following the pattern of `_fit_econml_dml`:
   - Accept `(df: pd.DataFrame, config: CausalModelConfig) -> Any`
   - Call `_prepare_data(df, config)` to get Y, T, X, W
   - Use `_encode_categoricals` for any additional categorical handling
   - Log the ATE estimate with `logger.info`
   - Return the fitted estimator object

3. **Add a branch** in `train_causal_model` and `estimate_ate` for the new key.

4. **Add a test** in `tests/test_causal_model.py`:
   - Test that the estimator returns a model with `.effect()` method
   - Test that `estimate_ate` returns a positive float in [0.05, 0.50]

5. **Update `config/config.yaml`** if you want it as the default.

Key constraints:
- Always use `_prepare_data` — it handles anomaly filtering and categorical encoding consistently
- Seed random state with `random_state=42`
- The model must support `.effect(X=X_array)` for CATE inversion in `control.py`
