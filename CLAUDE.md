# CLAUDE.md — process_control_causal_ml

Autonomous Process Control with Causal ML — implementation guide for Claude Code.

---

## Project Overview

Simulate a continuous chemical production line, learn its causal structure, detect deviations, and recommend corrective actions. Fully runnable on 2-CPU GitHub Codespaces (no GPU).

**Stack:** Python 3.11+, uv, src/ layout, Pydantic v2 config, loguru, Great Expectations, pytest, FastAPI, Plotly Dash, GitHub Actions CI.

---

## Repository Structure

```
process_control_causal_ml/
├── .claude/skills/
│   ├── add-anomaly-type.md
│   ├── add-estimator.md
│   ├── check-quality.md
│   └── run-pipeline.md
├── .github/workflows/ci.yml
├── .devcontainer/devcontainer.json
├── config/config.yaml
├── data/.gitkeep
├── notebooks/exploration.ipynb
├── src/process_control_causal_ml/
│   ├── __init__.py
│   ├── simulate.py
│   ├── causal_graph.py
│   ├── causal_model.py
│   ├── detect.py
│   ├── control.py
│   ├── serve.py
│   ├── dashboard.py
│   └── utils.py
├── tests/
│   ├── conftest.py
│   ├── test_simulate.py
│   ├── test_causal_graph.py
│   ├── test_causal_model.py
│   ├── test_detect.py
│   ├── test_control.py
│   ├── test_serve.py
│   └── test_utils.py
├── great_expectations/expectations/process_data_suite.json
├── Makefile
├── pyproject.toml
└── README.md
```

---

## Implementation Order

Build modules in dependency order. Do not skip ahead.

1. `pyproject.toml` + `uv` setup
2. `config/config.yaml` + Pydantic config models in `utils.py`
3. `simulate.py` — SCM data generation
4. `causal_graph.py` — DAG discovery
5. `causal_model.py` — treatment effect estimation
6. `detect.py` — anomaly detection
7. `control.py` — corrective action recommender
8. `serve.py` — FastAPI app
9. `dashboard.py` — Plotly Dash monitoring UI
10. Tests for each module
11. `Makefile`, CI, Great Expectations suite

---

## Module Specifications

### `config/config.yaml`

Single source of truth. Validated by Pydantic v2 at startup.

```yaml
simulation:
  n_batches: 50000
  anomaly_fraction: 0.05
  random_seed: 42
  catalyst_types: ["A", "B", "C"]
  catalyst_temp_effects: {"A": 0.0, "B": 2.5, "C": -1.5}

causal_graph:
  method: "pc"                     # "pc" | "lingam" | "ges"
  significance_level: 0.05
  max_cond_vars: 3

causal_model:
  estimator: "econml_dml"          # "dowhy_linear" | "econml_dml" | "econml_causal_forest"
  treatment: "reactor_temp"
  outcome: "product_yield"
  common_causes:
    - "catalyst_type"
    - "coolant_flow_rate"
    - "ph_level"
  effect_modifiers:
    - "catalyst_type"

detection:
  isolation_forest_contamination: 0.05
  cusum_threshold: 5.0
  cusum_drift: 0.5
  window_size: 20

control:
  target_product_yield: 87.0
  target_tolerance: 1.0
  max_temp_adjustment: 5.0
  max_cooling_adjustment: 10.0

serving:
  host: "0.0.0.0"
  port: 8000
```

### `utils.py`

- Load and validate `config.yaml` using Pydantic v2 `BaseModel` with nested models for each section.
- Provide `load_config(path: str) -> AppConfig`.
- Configure `loguru` logger exported as `logger`.
- All other modules import `logger` and `load_config` from here.

### `simulate.py`

Implement the SCM exactly as specified. Key functions:

```python
def generate_process_data(config: SimulationConfig) -> pd.DataFrame
def inject_anomalies(df: pd.DataFrame, config: SimulationConfig) -> pd.DataFrame
def validate_data(df: pd.DataFrame) -> None   # Great Expectations
```

**SCM equations (implement precisely):**

```
catalyst_type ~ Categorical([A, B, C], p=[0.4, 0.35, 0.25])
coolant_flow_rate ~ Normal(mu=50, sigma=5)

reactor_temp = 180.0
    + catalyst_effect[catalyst_type]    # {A: 0.0, B: 2.5, C: -1.5}
    - 0.20 * coolant_flow_rate
    + noise(sigma=1.0)

pressure = 2.5
    + 0.06 * reactor_temp
    + noise(sigma=0.15)

ph_level = 7.0
    - 0.05 * pressure
    - 0.02 * coolant_flow_rate
    + noise(sigma=0.06)

reaction_rate = 12.0
    + 0.35 * reactor_temp
    + interaction_effect(reactor_temp, catalyst_type)  # {A: 0.0, B: 0.05*rt, C: -0.03*rt}
    - 0.6 * ph_level
    + noise(sigma=0.6)

product_yield = 85.0
    + 0.30 * reaction_rate
    + 0.08 * reactor_temp
    - 0.40 * pressure
    + noise(sigma=0.25)
```

**Anomaly injection** (~5% of batches, three types):
- `drift`: gradual mean shift of +10°C over 50 consecutive batches on `reactor_temp`
- `step_change`: sudden +3σ jump on `pressure`
- `sensor_noise`: variance spike ×5 on `ph_level`

Include `batch_id` (int), `timestamp` (datetime, 1 batch per hour from 2023-01-01), `anomaly_flag` (bool), `anomaly_type` (str, "none"/"drift"/"step_change"/"sensor_noise").

Output: `data/process_data.parquet`

### `causal_graph.py`

```python
def discover_dag(df: pd.DataFrame, config: CausalGraphConfig) -> nx.DiGraph
def compare_to_ground_truth(learned: nx.DiGraph, true_dag: nx.DiGraph) -> dict
def plot_dag(dag: nx.DiGraph, path: str) -> None
def get_ground_truth_dag() -> nx.DiGraph
```

**Ground truth DAG edges:**
```
catalyst_type -> reactor_temp
coolant_flow_rate -> reactor_temp
coolant_flow_rate -> ph_level
reactor_temp -> pressure
reactor_temp -> reaction_rate
reactor_temp -> product_yield
pressure -> reaction_rate
pressure -> ph_level
ph_level -> reaction_rate
ph_level -> product_yield
reaction_rate -> product_yield
```

**Discovery methods:**
- PC algorithm via `causal-learn` (`causallearn.search.ConstraintBased.PC`)
- LiNGAM via `lingam.DirectLiNGAM`
- GES via `causallearn.search.ScoreBased.GES`

**Important:** Subsample to 5,000 rows for graph discovery (performance). Use full dataset for effect estimation.

One-hot encode `catalyst_type` before passing to discovery algorithms. After discovery, map nodes back to original variable names.

Evaluation: compute **Structural Hamming Distance (SHD)** — count of missing edges + extra edges + reversed edges vs ground truth.

Output: `data/causal_graph.pkl` (nx.DiGraph), `data/causal_graph.png`

### `causal_model.py`

```python
def estimate_ate(df: pd.DataFrame, config: CausalModelConfig) -> float
def estimate_cate(df: pd.DataFrame, config: CausalModelConfig) -> pd.Series
def estimate_interaction_effects(df: pd.DataFrame) -> pd.DataFrame
def refute_estimate(model, estimate, n_simulations: int = 100) -> dict
def train_causal_model(df: pd.DataFrame, config: CausalModelConfig) -> object
```

**Estimators to implement:**

1. `dowhy_linear` — DoWhy with linear regression estimator. Use `dowhy.CausalModel`. Common causes from config. Run 3 refutation tests.

2. `econml_dml` — EconML `LinearDML`. Treatment: `reactor_temp` (continuous). Outcome: `product_yield`. Confounders: one-hot encoded `catalyst_type` + `coolant_flow_rate` + `ph_level`. Effect modifiers: one-hot encoded `catalyst_type`.

3. `econml_causal_forest` — EconML `CausalForestDML`. Same setup. Returns heterogeneous CATE per observation.

**Categorical handling:** Always one-hot encode `catalyst_type` before passing to EconML. For DoWhy use string labels directly.

**Interaction effects table:** For each catalyst type (A/B/C), compute mean CATE from the causal forest — this quantifies `reactor_temp × catalyst_type` interaction.

Output: `data/causal_model.pkl`, `data/ate_results.json`

### `detect.py`

```python
@dataclass
class AnomalyResult:
    flag: bool
    score: float
    type: str       # "isolation_forest" | "cusum" | "none"
    variable: str   # most anomalous variable or "multivariate"

def train_detector(df: pd.DataFrame, config: DetectionConfig) -> AnomalyDetector
def detect_anomaly(reading: dict, detector: AnomalyDetector) -> AnomalyResult
def run_cusum(series: pd.Series, config: DetectionConfig) -> pd.Series
```

**Two-layer detection:**

Layer 1 — Isolation Forest:
- Train on normal data (`anomaly_flag == False`)
- Features: all continuous process variables
- If `score < threshold` → anomaly flagged

Layer 2 — CUSUM per variable:
- Maintain running CUSUM statistic for each sensor variable
- Detect mean shifts using `cusum_threshold` and `cusum_drift` from config
- Identify which variable triggered CUSUM → set `AnomalyResult.variable`

`AnomalyDetector` is a dataclass holding the fitted IsolationForest, per-variable CUSUM state, feature scaler, and config.

### `control.py`

```python
@dataclass
class CorrectiveAction:
    variable: str
    current: float
    recommended: float
    delta: float
    confidence: float   # width of 95% CI from CATE model

def recommend_action(
    anomaly: AnomalyResult,
    causal_model,        # fitted EconML model
    current_state: dict,
    config: ControlConfig
) -> CorrectiveAction
```

**Algorithm:**
1. If no anomaly, return no-op action.
2. Identify root-cause variable from `anomaly.variable` and causal DAG (upstream controllable variables are `reactor_temp` or `coolant_flow_rate`).
3. Use CATE model to compute: required `delta_yield = target - current_yield`.
4. Invert: `delta_treatment = delta_yield / cate_estimate`.
5. Clip to safe bounds from config (`max_temp_adjustment`, `max_cooling_adjustment`).
6. Compute confidence from CATE model's `effect_interval()`.
7. Return `CorrectiveAction`.

### `dashboard.py`

Plotly Dash monitoring UI. Launched by `make serve` (replaces the direct uvicorn command).

```python
def _load_artefacts() -> dict   # loads all pkl/json/png files from data/ at startup
def main() -> None              # creates and runs the Dash app
```

**Four tabs:**
1. **Process Monitor** — `dcc.Dropdown` (multi-select variables), anomaly highlight toggle, time-series subplots via `make_subplots`
2. **Causal Graph** — side-by-side learned vs ground-truth DAG images, SHD metrics table, explanation card
3. **Causal Effects** — ATE summary card, CATE-by-catalyst bar chart with `add_hline` reference, interpretation guide
4. **Live Inference** — `dcc.Slider` per sensor, `dbc.RadioItems` for catalyst_type, 4-step algorithm explainer card, real-time anomaly detection + corrective action output

**Pipeline banner** at the top: 5 `dbc.Card` components showing pipeline stages (simulate → discover → estimate → detect → control).

Use `dash-bootstrap-components` (dbc) for layout. Load all artefacts once at startup, not per callback. Bind to `config.serving.host` and `config.serving.port`.

### `serve.py`

FastAPI app. Load config, models, and detector at startup using `lifespan` context manager. Do not reload on each request.

**Important:** `make serve` now launches `dashboard.py`, not uvicorn directly. `serve.py` is still used for the REST API (testable via `httpx.TestClient`) but is not the primary serving target.

```
POST /predict
  Body:  ProcessReading (Pydantic model with all sensor fields)
  Returns: PredictResponse {
    anomaly_flag: bool,
    anomaly_score: float,
    anomaly_type: str,
    corrective_action: {variable, current, recommended, delta, confidence}
  }

GET /health
  Returns: {"status": "ok", "model_loaded": bool}

GET /causal_graph
  Returns: {"image_base64": "<base64 PNG>"}

GET /ate
  Returns: {"ate": float, "treatment": str, "outcome": str}
```

Use Pydantic v2 models for all request/response schemas. Log every request with loguru.

---

## `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "process-control-causal-ml"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.26",
    "scikit-learn>=1.4",
    "econml>=0.15",
    "dowhy>=0.11",
    "causal-learn>=0.1.3",
    "lingam>=1.8",
    "pgmpy>=0.1.25",
    "networkx>=3.2",
    "matplotlib>=3.8",
    "fastapi>=0.110",
    "uvicorn>=0.29",
    "pydantic>=2.6",
    "loguru>=0.7",
    "great-expectations>=0.18",
    "pyarrow>=15.0",
    "pyyaml>=6.0",
    "dash>=2.16",
    "dash-bootstrap-components>=1.6",
    "plotly>=5.20",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "ruff>=0.4",
    "mypy>=1.9",
    "httpx>=0.27",   # FastAPI test client
]

[tool.hatch.build.targets.wheel]
packages = ["src/process_control_causal_ml"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--tb=short"
```

---

## Makefile

Organized into five sections with inline `##` descriptions.

```makefile
# Setup
install:          ## Install all dependencies (including dev extras)
	uv sync --extra dev

# Pipeline
simulate:         ## Generate synthetic process data via SCM
	uv run python -m process_control_causal_ml.simulate

validate:         ## Validate existing process data against schema and ranges
	uv run python -m process_control_causal_ml.simulate --validate-only

graph:            ## Discover causal DAG from process data
	uv run python -m process_control_causal_ml.causal_graph

train:            ## Train causal model and anomaly detector
	uv run python -m process_control_causal_ml.causal_model
	uv run python -m process_control_causal_ml.detect --train

evaluate:         ## Evaluate DAG quality (SHD) and causal model (refutation tests)
	uv run python -m process_control_causal_ml.causal_graph --evaluate
	uv run python -m process_control_causal_ml.causal_model --refute

serve:            ## Start Plotly Dash dashboard on port 8000
	uv run python -m process_control_causal_ml.dashboard

all: simulate validate graph train evaluate  ## Run full pipeline end-to-end

# Testing
test:             ## Run pytest with coverage report
	uv run pytest tests/ -v --cov=src/process_control_causal_ml --cov-report=term-missing

# Code quality
lint:             ## Auto-fix style issues, format code, and run type checker
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/
	uv run ty check src/ tests/

complexity:       ## Cyclomatic complexity report (B-grade and below only)
	uv run radon cc src/ -a -s -nb

maintainability:  ## Maintainability index report per module
	uv run radon mi src/ -s

# Security
audit:            ## Dependency vulnerability audit
	uv run pip-audit --skip-editable

security:         ## Static security analysis (medium and high severity)
	uv run bandit -r src/ --severity-level medium -q
```

Each module must be runnable as `python -m process_control_causal_ml.<module>` with a `if __name__ == "__main__":` block that invokes the relevant pipeline step.

---

## GitHub Actions CI (`.github/workflows/ci.yml`)

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
        with:
          python-version: "3.11"
      - run: uv sync --extra dev
      - run: make lint
      - run: make test
```

---

## `.devcontainer/devcontainer.json`

```json
{
  "name": "process-control-causal-ml",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "features": {
    "ghcr.io/devcontainers/features/python:1": {}
  },
  "postCreateCommand": "pip install uv && uv sync --extra dev",
  "customizations": {
    "vscode": {
      "extensions": ["ms-python.python", "ms-python.ruff"]
    }
  }
}
```

---

## Great Expectations Suite (`great_expectations/expectations/process_data_suite.json`)

Validate `data/process_data.parquet` with these expectations:
- All required columns present
- `product_yield` in range [50, 120]
- `reactor_temp` in range [150, 210]
- `pressure` in range [2.0, 4.0]
- `ph_level` in range [5.5, 8.5]
- `catalyst_type` values in {"A", "B", "C"}
- `anomaly_fraction` between 0.03 and 0.08
- No nulls in any column

---

## Tests

102 tests, ~67% coverage. Each test file uses `pytest` with fixtures defined in `conftest.py`.

### `tests/conftest.py`
- `small_df` fixture: generate 500 rows using `generate_process_data` with `random_seed=0`
- `config` fixture: load default config

### `tests/test_simulate.py`
- Test output columns match expected schema
- Test anomaly fraction is within [0.03, 0.08]
- Test SCM equations: regress `reactor_temp` on `coolant_flow_rate`, verify negative coefficient
- Test `inject_anomalies` does not alter non-anomaly rows

### `tests/test_causal_graph.py`
- Test `discover_dag` returns `nx.DiGraph` with correct node count
- Test `get_ground_truth_dag` has 11 edges
- Test `compare_to_ground_truth` returns dict with keys `shd`, `precision`, `recall`
- Tests for internal helpers: `_encode_for_discovery`, `_build_dag_from_adjacency`, `_node_name_to_col`, `_map_dag_to_original_names`, `_extract_dag_from_causal_learn`
- LiNGAM and GES discovery (GES marked `xfail` due to causal-learn/numpy scalar bug)
- Extra/reversed edge comparison tests

### `tests/test_causal_model.py`
- Test `estimate_ate` returns a float
- Test ATE sign is positive (higher `reactor_temp` → higher `product_yield`)
- Test `estimate_interaction_effects` returns DataFrame with 3 rows (one per catalyst type)
- Test `_encode_categoricals` one-hot output and numeric passthrough
- Test anomaly row exclusion in `_prepare_data`
- Test `train_causal_model` returns model with `.effect()` method
- Test unknown estimator raises `ValueError`

### `tests/test_detect.py`
- Test `train_detector` returns `AnomalyDetector` with fitted model
- Test `detect_anomaly` returns `AnomalyResult` with correct types
- Test `run_cusum` returns a Series of same length as input
- Test constant series CUSUM returns all zeros
- Test isolation_forest detection type and finite scores
- Test CUSUM alarm triggering with large shift
- Test `iso_threshold` is negative (IsolationForest convention)

### `tests/test_control.py`
- Test `recommend_action` returns `CorrectiveAction` with `delta` within config bounds
- Test no-op when `anomaly.flag == False`
- Test all anomaly variable routing (reactor_temp, pressure, ph_level → coolant_flow_rate)
- Test above-target yield produces non-positive delta
- Test DoWhy `.value` model path
- Test zero CATE → delta=0

### `tests/test_serve.py`
- `TestClient` fixture (module-scoped) with all models loaded
- Test `/health` 200 + schema
- Test `/predict` 200 with valid body, 422 with invalid body
- Test `/predict` 503 without models loaded (monkeypatch `_state.model_loaded`)
- Test `/causal_graph` 404 when PNG missing
- Test `/ate` 200 + schema

### `tests/test_utils.py`
- Test `load_config` defaults and YAML parsing
- Test `CausalGraphConfig` method validator
- Test `CausalModelConfig` estimator validator
- Test `SimulationConfig` defaults and `AppConfig` section access

**CI constraint:** Tests must pass in under 3 minutes on 2-core CPU. Use `n_batches=1000` and `n_simulations=10` in test fixtures.

---

## Engineering Rules

1. **No global state.** Load config and models explicitly; pass as arguments.
2. **Pydantic v2 everywhere.** All configs and API schemas use `model_validator` / `field_validator`, not v1 `@validator`.
3. **loguru for all logging.** No `print()` statements in src code.
4. **Paths always from config or CLI args.** No hardcoded paths except defaults in the CLI `__main__` block.
5. **Seed everything.** Pass `random_seed` from config to numpy, sklearn, and any other stochastic components.
6. **Subsample for graph discovery.** Always subsample to max 5,000 rows in `discover_dag`.
7. **One-hot encode categoricals** before passing to EconML/sklearn. Use `pd.get_dummies` with `drop_first=True` for causal graph discovery (prevents singular correlation matrix in PC algorithm). Use `drop_first=False` when passing to EconML estimators to preserve all levels for CATE computation.
8. **Persist artefacts to `data/`.** Each pipeline stage saves its output (parquet, pkl, json, png) to `data/`.
9. **All functions type-annotated.** mypy must pass with `ignore_missing_imports = true`.
10. **FastAPI startup only.** Load all models during `lifespan`, not per-request.
11. **Pickle module path fix for `detect.py`.** When running as `python -m`, dataclasses get `__module__ = '__main__'`. Patch both `sys.modules` and `__module__` on each dataclass at the bottom of `detect.py` to ensure correct deserialization after pickling:
    ```python
    if __spec__ is not None:
        _real_name = __spec__.name
        sys.modules.setdefault(_real_name, sys.modules[__name__])
        for _cls in (AnomalyResult, CusumState, AnomalyDetector):
            _cls.__module__ = _real_name
    ```
12. **Dashboard over direct uvicorn.** `make serve` launches `dashboard.py` (Plotly Dash), not uvicorn directly. The FastAPI `serve.py` is still available for REST API access and testing.

---

## Causal DAG (Reference)

```
catalyst_type ──► reactor_temp ──────────────────────────────► product_yield
                      │                                              ▲
                      ▼                                              │
coolant_flow_rate ──► pressure ──► reaction_rate ───────────────────┘
      │                   │              ▲
      │                   ▼              │
      └──────────────► ph_level ─────────┘
```

Ground-truth ATE of `reactor_temp` on `product_yield`:
- Direct effect: +0.08 per °C
- Indirect via `reaction_rate`: +0.30 × (0.35) = +0.105 per °C
- Indirect via `pressure → reaction_rate`: small negative offset
- **Expected total ATE ≈ +0.17 to +0.20 per °C**

---

## Claude Skills (`.claude/skills/`)

Reusable task guides for common extension patterns:

| Skill file | Invoked when |
|---|---|
| `run-pipeline.md` | User asks to run, retrain, or serve |
| `check-quality.md` | User asks to check quality, lint, or run tests |
| `add-estimator.md` | User asks to add a new causal effect estimator |
| `add-anomaly-type.md` | User asks to add a new anomaly injection type |

---

## Evaluation Targets

| Component | Metric | Target |
|---|---|---|
| Causal graph | SHD vs ground truth | ≤ 3 |
| ATE estimation | Bias vs ground truth (≈0.185) | < 10% |
| Anomaly detection | F1 score | ≥ 0.80 |
| Corrective action | % batches restored to spec | ≥ 75% |
| API latency | p95 response time (`/predict`) | < 200 ms |
| Test suite | Coverage | ≥ 65% |
