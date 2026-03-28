---
description: Add a new anomaly type to the simulation
---

To add a new anomaly injection type to `simulate.py`:

1. **Write a `_inject_<type>` helper** following the pattern of `_inject_drift`, `_inject_step_change`, `_inject_sensor_noise`:
   ```python
   def _inject_<type>(
       df: pd.DataFrame,
       rng: np.random.Generator,
       n: int,
       budget: int,
       used: set[int],
   ) -> None:
       """Inject <description> (in-place)."""
       available = [i for i in range(n) if i not in used]
       indices = rng.choice(available, size=min(budget, len(available)), replace=False)
       for idx in indices:
           df.at[idx, "<variable>"] = <modified_value>
           df.at[idx, "anomaly_flag"] = True
           df.at[idx, "anomaly_type"] = "<type_name>"
           used.add(idx)
   ```

2. **Call it** from `inject_anomalies` after the existing three calls, passing the shared `used_indices` set.

3. **Update `validate_data`** — add the new type string to the `anomaly_type` check if needed.

4. **Update tests** in `test_simulate.py`:
   - Add the new type to `test_anomaly_types`
   - Add `test_inject_anomalies_<type>_type_present`

5. **Update the detection** in `detect.py` if the new anomaly targets a variable not currently in `FEATURE_COLS`.

Key constraints:
- Use the shared `used_indices` set to avoid double-labelling rows
- Budget = `anomaly_budget` (= `int(n * anomaly_fraction) // 3`) unless you want a different fraction
- The anomaly_type string must be one word, lowercase with underscores
