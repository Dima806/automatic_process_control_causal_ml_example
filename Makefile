.PHONY: simulate validate graph train evaluate serve test lint audit security complexity maintainability install all

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

install:          ## Install all dependencies (including dev extras)
	uv sync --extra dev

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:             ## Run pytest with coverage report
	uv run pytest tests/ -v --cov=src/process_control_causal_ml --cov-report=term-missing

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint:             ## Auto-fix style issues, format code, and run type checker
	uv run ruff check --fix src/ tests/
	uv run ruff format src/ tests/
	uv run ty check src/ tests/

complexity:       ## Cyclomatic complexity report (B-grade and below only)
	uv run radon cc src/ -a -s -nb

maintainability:  ## Maintainability index report per module
	uv run radon mi src/ -s

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

audit:            ## Dependency vulnerability audit (skips local packages)
	# --skip-editable: skip local packages not on PyPI (this project itself)
	# --ignore-vuln CVE-2026-4539: pygments 2.19.2 is the latest release; no upstream fix available yet
	uv run pip-audit --skip-editable --ignore-vuln CVE-2026-4539

security:         ## Static security analysis (medium and high severity)
	uv run bandit -r src/ --severity-level medium -q
