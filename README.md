# Autonomous Process Control with Causal ML

A production-grade Python pipeline that simulates a continuous chemical manufacturing process,
learns its causal structure from observational data, detects process deviations, and recommends
corrective parameter adjustments — fully runnable on 2-CPU GitHub Codespaces.

---

## Pipeline Overview

```
catalyst_type ──► reactor_temp ──────────────────────────────► product_yield
                      │                                              ▲
                      ▼                                              │
coolant_flow_rate ──► pressure ──► reaction_rate ───────────────────┘
      │                   │              ▲
      │                   ▼              │
      └──────────────► ph_level ─────────┘
```

| Stage | Module | What it does |
|---|---|---|
| Simulate | `simulate.py` | Generate synthetic data from a Structural Causal Model (SCM) |
| Graph | `causal_graph.py` | Discover the causal DAG via PC / LiNGAM |
| Train | `causal_model.py` | Estimate ATE + CATE with EconML LinearDML / CausalForestDML |
| Detect | `detect.py` | Flag deviations — Isolation Forest + per-variable CUSUM |
| Control | `control.py` | Recommend minimal corrective adjustments via CATE inversion |
| Serve | `serve.py` | FastAPI REST endpoints (`/predict`, `/health`, `/ate`, `/causal_graph`) |
| Dashboard | `dashboard.py` | Plotly Dash monitoring UI with live inference tab |

---

## Quick Start

```bash
# Install dependencies
make install

# Run the full pipeline
make all          # simulate → validate → graph → train → evaluate

# Launch the dashboard
make serve        # http://127.0.0.1:8000
```

---

## Makefile Reference

### Pipeline

| Target | Description |
|---|---|
| `make simulate` | Generate 50,000 synthetic process batches → `data/process_data.parquet` |
| `make validate` | Validate existing data schema and value ranges |
| `make graph` | Discover causal DAG → `data/causal_graph.pkl` + `data/causal_graph.png` |
| `make train` | Train causal model + anomaly detector → `data/causal_model.pkl`, `data/detector.pkl` |
| `make evaluate` | Compute SHD, DAG metrics, and refutation tests → `data/dag_metrics.json` |
| `make serve` | Start Plotly Dash dashboard on port 8000 |
| `make all` | Run full pipeline end-to-end |

### Testing & Quality

| Target | Description |
|---|---|
| `make test` | pytest with coverage report |
| `make lint` | ruff check + format + ty type check |
| `make complexity` | Radon cyclomatic complexity (B-grade and below) |
| `make maintainability` | Radon maintainability index per module |

### Security

| Target | Description |
|---|---|
| `make audit` | pip-audit dependency vulnerability scan |
| `make security` | Bandit static analysis (medium + high severity) |

---

## Architecture

### Structural Causal Model (SCM)

The simulation implements exact SCM equations so the ground-truth causal structure is known:

```
reactor_temp    = 180 + catalyst_effect − 0.20 × coolant_flow_rate + ε(1.0)
pressure        = 2.5 + 0.06 × reactor_temp + ε(0.15)
ph_level        = 7.0 − 0.05 × pressure − 0.02 × coolant_flow_rate + ε(0.06)
reaction_rate   = 12  + 0.35 × reactor_temp + interaction(rt, catalyst) − 0.6 × ph_level + ε(0.6)
product_yield   = 85  + 0.30 × reaction_rate + 0.08 × reactor_temp − 0.40 × pressure + ε(0.25)
```

Expected ATE of `reactor_temp` → `product_yield` ≈ **+0.17 to +0.20 per °C**.

### Anomaly Types

| Type | Variable | Description |
|---|---|---|
| `drift` | `reactor_temp` | Gradual +10 °C shift over 50 consecutive batches |
| `step_change` | `pressure` | Sudden +3σ spike |
| `sensor_noise` | `ph_level` | Variance spike ×5 |

### Causal Effect Estimation

- **`dowhy_linear`** — DoWhy with linear regression + 3 refutation tests
- **`econml_dml`** — EconML LinearDML (Double Machine Learning, default)
- **`econml_causal_forest`** — EconML CausalForestDML for heterogeneous CATE

### Two-Layer Anomaly Detection

**Layer 1 — Isolation Forest**: multivariate global score. Points below `iso_threshold` are flagged.
**Layer 2 — CUSUM per variable**: sequential mean-shift detector. Identifies *which* sensor triggered.

### Corrective Action

1. Map anomalous variable to controllable upstream input via the causal DAG
2. Compute required yield gap: `Δyield = target − current`
3. Invert via CATE: `Δtreatment = Δyield / cate_estimate`
4. Clip to safe bounds (`max_temp_adjustment`, `max_cooling_adjustment`)
5. Report confidence from 95% CATE CI width

---

## REST API (`serve.py`)

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Model load status |
| `/predict` | POST | Anomaly detection + corrective action for a sensor reading |
| `/ate` | GET | Average Treatment Effect summary |
| `/causal_graph` | GET | Base64-encoded causal DAG PNG |

**Example:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "catalyst_type": "B",
    "coolant_flow_rate": 50.0,
    "reactor_temp": 183.0,
    "pressure": 2.75,
    "ph_level": 6.9,
    "reaction_rate": 74.0,
    "product_yield": 84.0
  }'
```

---

## Dashboard (`dashboard.py`)

Four tabs available at `http://127.0.0.1:8000` after `make serve`:

| Tab | Content |
|---|---|
| **Process Monitor** | Time-series of sensor variables; toggle anomaly highlighting |
| **Causal Graph** | Learned DAG vs ground-truth; SHD / precision / recall table |
| **Causal Effects** | ATE card; CATE-by-catalyst bar chart with interpretation guide |
| **Live Inference** | Sliders for all sensors; real-time anomaly detection + corrective action |

---

## Evaluation Targets

| Component | Metric | Target |
|---|---|---|
| Causal graph | SHD vs ground truth | ≤ 3 |
| ATE estimation | Bias vs ground truth (≈0.185) | < 10% |
| Anomaly detection | F1 score | ≥ 0.80 |
| Corrective action | Batches restored to spec | ≥ 75% |
| API latency | p95 response time | < 200 ms |

---

## Stack

- **Python 3.11+**, `uv`, `src/` layout, `hatchling`
- **Causal**: `causal-learn`, `lingam`, `dowhy`, `econml`
- **ML**: `scikit-learn`, `numpy`, `pandas`
- **API**: `fastapi`, `uvicorn`, `pydantic v2`
- **Dashboard**: `dash`, `dash-bootstrap-components`, `plotly`
- **Quality**: `ruff`, `ty`, `radon`, `bandit`, `pip-audit`
- **Testing**: `pytest`, `pytest-cov`
- **Config**: `pyyaml`, `loguru`

---

## Project Structure

```
├── config/config.yaml              Single config source (Pydantic-validated)
├── data/                           Pipeline artefacts (gitignored except .gitkeep)
├── notebooks/exploration.ipynb     End-to-end walkthrough
├── src/process_control_causal_ml/
│   ├── simulate.py                 SCM data generation + anomaly injection
│   ├── causal_graph.py             DAG discovery (PC / LiNGAM / GES)
│   ├── causal_model.py             Treatment effect estimation
│   ├── detect.py                   Anomaly detection (IsolationForest + CUSUM)
│   ├── control.py                  Corrective action recommender
│   ├── serve.py                    FastAPI REST API
│   ├── dashboard.py                Plotly Dash dashboard
│   └── utils.py                    Config models + logger
├── tests/                          pytest test suite (102 tests, 67% coverage)
├── Makefile
└── pyproject.toml
```
