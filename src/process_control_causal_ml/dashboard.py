"""Plotly Dash monitoring dashboard for the process control pipeline."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import dash
import dash_bootstrap_components as dbc
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from plotly.subplots import make_subplots

from process_control_causal_ml.control import recommend_action
from process_control_causal_ml.detect import detect_anomaly
from process_control_causal_ml.utils import load_config, logger

DATA_DIR = Path("data")

SENSOR_VARS = [
    "reactor_temp",
    "product_yield",
    "pressure",
    "ph_level",
    "reaction_rate",
    "coolant_flow_rate",
]

# ---------------------------------------------------------------------------
# Load artefacts at startup
# ---------------------------------------------------------------------------


def _load_artefacts() -> dict[str, Any]:
    arts: dict[str, Any] = {}

    parquet_path = DATA_DIR / "process_data.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        arts["df"] = df.sample(n=min(2000, len(df)), random_state=42).sort_values("batch_id")
    else:
        arts["df"] = None
        logger.warning("process_data.parquet not found — run 'make simulate'")

    model_path = DATA_DIR / "causal_model.pkl"
    if model_path.exists():
        saved = joblib.load(model_path)
        arts["causal_model"] = saved["model"]
    else:
        arts["causal_model"] = None

    detector_path = DATA_DIR / "detector.pkl"
    if detector_path.exists():
        arts["detector"] = joblib.load(detector_path)
    else:
        arts["detector"] = None

    ate_path = DATA_DIR / "ate_results.json"
    if ate_path.exists():
        with open(ate_path) as f:
            arts["ate_results"] = json.load(f)
    else:
        arts["ate_results"] = None

    dag_metrics_path = DATA_DIR / "dag_metrics.json"
    if dag_metrics_path.exists():
        with open(dag_metrics_path) as f:
            arts["dag_metrics"] = json.load(f)
    else:
        arts["dag_metrics"] = None

    def _img_b64(path: Path) -> str | None:
        return base64.b64encode(path.read_bytes()).decode() if path.exists() else None

    arts["learned_dag_b64"] = _img_b64(DATA_DIR / "causal_graph.png")
    arts["true_dag_b64"] = _img_b64(DATA_DIR / "causal_graph_true.png")
    arts["config"] = load_config()
    return arts


_arts = _load_artefacts()

# ---------------------------------------------------------------------------
# Tab builders
# ---------------------------------------------------------------------------


def _process_monitor_tab() -> dbc.Tab:
    df = _arts.get("df")
    options = (
        [{"label": v.replace("_", " ").title(), "value": v} for v in SENSOR_VARS if v in df.columns]
        if df is not None
        else []
    )
    return dbc.Tab(
        label="Process Monitor",
        tab_id="tab-monitor",
        children=[
            dbc.Row(
                dbc.Col(
                    [
                        html.H5("Process Variable Time Series", className="mt-3 mb-1"),
                        html.P(
                            [
                                "Time series of 2,000 sampled batches. ",
                                html.Span("Blue", style={"color": "#3498db", "fontWeight": "bold"}),
                                " = normal operation. ",
                                html.Span(
                                    "Red ×", style={"color": "#e74c3c", "fontWeight": "bold"}
                                ),
                                " = injected anomaly (drift on reactor_temp, step change on "
                                "pressure, sensor noise on ph_level). "
                                "Select variables below to compare; toggle the switch to show or hide anomaly markers.",
                            ],
                            className="text-muted small mb-2",
                        ),
                    ]
                )
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            id="sensor-var-dropdown",
                            options=options,
                            value=["reactor_temp", "product_yield"],
                            multi=True,
                            clearable=False,
                        ),
                        width=8,
                    ),
                    dbc.Col(
                        dbc.Checklist(
                            id="show-anomalies-check",
                            options=[{"label": "Highlight anomalies", "value": "show"}],
                            value=["show"],
                            switch=True,
                        ),
                        width=4,
                        className="d-flex align-items-center",
                    ),
                ],
                className="mb-3",
            ),
            dcc.Graph(id="process-timeseries"),
        ],
    )


def _causal_graph_tab() -> dbc.Tab:
    def _img_card(b64: str | None, title: str) -> dbc.Card:
        body = (
            html.Img(src=f"data:image/png;base64,{b64}", style={"width": "100%"})
            if b64
            else dbc.Alert("Image not found. Run 'make evaluate'.", color="warning")
        )
        return dbc.Card([dbc.CardHeader(title), dbc.CardBody(body)])

    dag_metrics = _arts.get("dag_metrics")
    if dag_metrics:
        rows = [
            ("SHD", dag_metrics["shd"]),
            ("Precision", f"{dag_metrics['precision']:.3f}"),
            ("Recall", f"{dag_metrics['recall']:.3f}"),
            ("True positives", dag_metrics["true_positive_edges"]),
            ("Missing edges", dag_metrics["missing_edges"]),
            ("Extra edges", dag_metrics["extra_edges"]),
            ("Reversed edges", dag_metrics["reversed_edges"]),
        ]
        metrics_body: Any = dbc.Table(
            [
                html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")])),
                html.Tbody([html.Tr([html.Td(k), html.Td(html.Strong(v))]) for k, v in rows]),
            ],
            bordered=True,
            size="sm",
            className="mt-2",
            style={"maxWidth": "360px"},
        )
    else:
        metrics_body = dbc.Alert("Run 'make evaluate' to see DAG metrics.", color="secondary")

    shd_explanation = dbc.Card(
        dbc.CardBody(
            [
                html.H6("How to read these metrics", className="fw-bold mb-2"),
                html.Ul(
                    [
                        html.Li(
                            [
                                html.Strong("SHD (Structural Hamming Distance): "),
                                "Total errors vs ground truth = missing + extra + reversed edges. "
                                "Lower is better; target ≤ 3.",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("Precision: "),
                                "Fraction of learned edges that are correct. "
                                "High precision → few spurious edges.",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("Recall: "),
                                "Fraction of true edges recovered. High recall → few missed edges.",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("Node colours: "),
                                html.Span(
                                    "green", style={"color": "#2ecc71", "fontWeight": "bold"}
                                ),
                                " = exogenous inputs (controllable),  ",
                                html.Span("red", style={"color": "#e74c3c", "fontWeight": "bold"}),
                                " = outcome (product_yield),  ",
                                html.Span("blue", style={"color": "#3498db", "fontWeight": "bold"}),
                                " = intermediate variables.",
                            ]
                        ),
                    ],
                    className="small mb-0",
                ),
            ]
        ),
        className="mt-3 mb-3 shadow-sm",
    )

    return dbc.Tab(
        label="Causal Graph",
        tab_id="tab-graph",
        children=[
            dbc.Row(
                dbc.Col(
                    html.P(
                        "The PC algorithm discovers the Markov equivalence class (CPDAG) from "
                        "conditional independence tests on observational data. "
                        "catalyst_type is one-hot encoded with drop_first=True before discovery "
                        "to prevent a singular correlation matrix. "
                        "One-hot columns are collapsed back to catalyst_type after discovery.",
                        className="text-muted small mt-3 mb-2",
                    )
                )
            ),
            dbc.Row(
                [
                    dbc.Col(_img_card(_arts.get("learned_dag_b64"), "Learned DAG (PC)"), md=6),
                    dbc.Col(_img_card(_arts.get("true_dag_b64"), "Ground-Truth DAG"), md=6),
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [html.H5("DAG Evaluation Metrics", className="mt-4 mb-2"), metrics_body],
                        md=4,
                    ),
                    dbc.Col(shd_explanation, md=8),
                ]
            ),
        ],
    )


def _causal_effects_tab() -> dbc.Tab:
    ate_results = _arts.get("ate_results")

    if ate_results:
        ate_val: float = ate_results.get("ate", 0.0)
        treatment: str = ate_results.get("treatment", "reactor_temp")
        outcome: str = ate_results.get("outcome", "product_yield")
        estimator: str = ate_results.get("estimator", "")

        ate_card = dbc.Card(
            dbc.CardBody(
                [
                    html.H6("Average Treatment Effect", className="text-muted mb-1"),
                    html.H2(f"{ate_val:.4f}", className="text-primary mb-1"),
                    html.P(
                        f"{treatment} → {outcome}   |   {estimator}",
                        className="text-muted small mb-0",
                    ),
                ]
            ),
            className="mb-3 shadow-sm",
            style={"maxWidth": "360px"},
        )

        interactions = ate_results.get("interaction_effects", [])
        if interactions:
            cate_df = pd.DataFrame(interactions)
            cate_fig = px.bar(
                cate_df,
                x="catalyst_type",
                y="mean_cate",
                title="CATE by Catalyst Type (reactor_temp → product_yield)",
                labels={"catalyst_type": "Catalyst", "mean_cate": "Mean CATE"},
                color="catalyst_type",
                color_discrete_sequence=px.colors.qualitative.Set2,
                text_auto=".4f",
            )
            cate_fig.add_hline(
                y=ate_val,
                line_dash="dash",
                line_color="grey",
                annotation_text=f"ATE = {ate_val:.4f}",
                annotation_position="top right",
            )
            cate_fig.update_layout(showlegend=False, height=380)
            cate_content: Any = dcc.Graph(figure=cate_fig)
        else:
            cate_content = dbc.Alert("No interaction effects data.", color="secondary")
    else:
        ate_card = dbc.Alert(
            "Run 'make train' to generate causal effect estimates.", color="warning"
        )
        cate_content = html.Div()

    effects_explainer = dbc.Card(
        dbc.CardBody(
            [
                html.H6("Interpreting ATE and CATE", className="fw-bold mb-2"),
                html.Ul(
                    [
                        html.Li(
                            [
                                html.Strong("ATE (Average Treatment Effect): "),
                                "The population-average causal effect of raising reactor_temp by 1 °C "
                                "on product_yield (%). Estimated via EconML LinearDML "
                                "(Double Machine Learning). Ground truth ≈ +0.17–0.20 per °C.",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("CATE (Conditional ATE): "),
                                "How the effect varies by catalyst type — captures the "
                                "reactor_temp × catalyst_type interaction term encoded in the SCM. "
                                "Expected ordering: CATE(B) > CATE(A) > CATE(C).",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("Dashed line: "),
                                "Global ATE for reference. Points above = catalyst amplifies the "
                                "temperature effect; below = catalyst dampens it.",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("Usage in control: "),
                                "The controller divides the yield gap by the current CATE to get the "
                                "required temperature adjustment: Δtemp = Δyield / CATE.",
                            ]
                        ),
                    ],
                    className="small mb-0",
                ),
            ]
        ),
        className="shadow-sm",
    )

    return dbc.Tab(
        label="Causal Effects",
        tab_id="tab-effects",
        children=[
            dbc.Row(
                dbc.Col(
                    html.P(
                        "Causal effect of reactor_temp on product_yield, estimated by "
                        "EconML LinearDML controlling for catalyst_type, coolant_flow_rate, and ph_level. "
                        "Run 'make train' to populate these results.",
                        className="text-muted small mt-3 mb-2",
                    )
                )
            ),
            dbc.Row(
                [
                    dbc.Col(ate_card, md=4),
                    dbc.Col(effects_explainer, md=8),
                ],
                className="mb-3",
            ),
            dbc.Row(dbc.Col(cate_content, md=10)),
        ],
    )


_SLIDER_SPECS: list[tuple[str, str, float, float, float]] = [
    ("reactor_temp", "Reactor Temp (°C)", 150.0, 220.0, 0.5),
    ("coolant_flow_rate", "Coolant Flow (L/min)", 20.0, 80.0, 0.5),
    ("pressure", "Pressure (bar)", 1.5, 5.0, 0.05),
    ("ph_level", "pH Level", 4.0, 9.0, 0.1),
    ("reaction_rate", "Reaction Rate (mol/s)", 40.0, 120.0, 0.5),
    ("product_yield", "Product Yield (%)", 50.0, 130.0, 0.5),
]

_DEFAULT_STATE: dict[str, Any] = {
    "catalyst_type": "B",
    "coolant_flow_rate": 50.0,
    "reactor_temp": 185.0,
    "pressure": 2.8,
    "ph_level": 6.9,
    "reaction_rate": 75.0,
    "product_yield": 84.0,
}


def _live_inference_tab() -> dbc.Tab:
    rows = []
    for key, label, lo, hi, step in _SLIDER_SPECS:
        default = _DEFAULT_STATE[key]
        rows.append(
            dbc.Row(
                [
                    dbc.Col(html.Label(label, className="fw-semibold"), width=4),
                    dbc.Col(
                        dcc.Slider(
                            id=f"slider-{key}",
                            min=lo,
                            max=hi,
                            step=step,
                            value=default,
                            marks={lo: str(lo), hi: str(hi)},
                            tooltip={"placement": "bottom", "always_visible": True},
                        ),
                        width=8,
                    ),
                ],
                className="mb-3 align-items-center",
            )
        )

    catalyst_row = dbc.Row(
        [
            dbc.Col(html.Label("Catalyst Type", className="fw-semibold"), width=4),
            dbc.Col(
                dbc.RadioItems(
                    id="radio-catalyst",
                    options=[{"label": c, "value": c} for c in ["A", "B", "C"]],
                    value="B",
                    inline=True,
                ),
                width=8,
            ),
        ],
        className="mb-3 align-items-center",
    )

    inference_explainer = dbc.Card(
        dbc.CardBody(
            [
                html.H6("How detection & control works", className="fw-bold mb-2"),
                html.Ol(
                    [
                        html.Li(
                            [
                                html.Strong("Layer 1 — Isolation Forest: "),
                                "Scores the full feature vector globally. "
                                "Score below threshold → multivariate anomaly flagged.",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("Layer 2 — CUSUM: "),
                                "Maintains a running shift statistic per variable. "
                                "Identifies which sensor deviated from its training baseline.",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("Root-cause mapping: "),
                                "The anomalous variable is mapped to a controllable upstream input "
                                "(reactor_temp or coolant_flow_rate) via the causal DAG.",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("CATE inversion: "),
                                "Δcontrol = (target_yield − current_yield) / CATE, "
                                "clipped to safe operating bounds.",
                            ]
                        ),
                    ],
                    className="small mb-0",
                ),
            ]
        ),
        className="shadow-sm mb-3",
    )

    return dbc.Tab(
        label="Live Inference",
        tab_id="tab-inference",
        children=[
            dbc.Row(dbc.Col(inference_explainer), className="mt-3"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H5("Set Process State", className="mb-2"),
                            html.P(
                                "Adjust sliders to simulate a sensor reading, then click the "
                                "button to run anomaly detection and get a corrective action.",
                                className="text-muted small mb-3",
                            ),
                            catalyst_row,
                            *rows,
                            dbc.Button(
                                "Run Detection & Control",
                                id="predict-btn",
                                color="primary",
                                className="mt-1",
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            html.H5("Detection & Control Results", className="mb-2"),
                            html.P(
                                [
                                    html.Span(
                                        "Green",
                                        style={"color": "#198754", "fontWeight": "bold"},
                                    ),
                                    " = normal.  ",
                                    html.Span(
                                        "Red",
                                        style={"color": "#dc3545", "fontWeight": "bold"},
                                    ),
                                    " = anomaly detected with recommended corrective action.",
                                ],
                                className="text-muted small mb-3",
                            ),
                            html.Div(id="inference-output"),
                        ],
                        md=6,
                    ),
                ]
            ),
        ],
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Process Control — Causal ML",
)

_PIPELINE_STEPS = [
    ("1. Simulate", "SCM generates synthetic process data with known causal structure."),
    ("2. Discover", "PC algorithm recovers the causal DAG from observational data."),
    ("3. Estimate", "EconML LinearDML estimates the causal effect of reactor_temp on yield."),
    ("4. Detect", "Isolation Forest + CUSUM flag deviations in real time."),
    ("5. Control", "CATE inversion recommends the minimal corrective parameter adjustment."),
]

_pipeline_banner = dbc.Row(
    [
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    [
                        html.Span(step, className="fw-bold d-block small text-primary"),
                        html.Span(desc, className="small text-muted"),
                    ],
                    className="p-2",
                ),
                className="h-100 shadow-sm",
            ),
            xs=12,
            sm=6,
            md=True,
        )
        for step, desc in _PIPELINE_STEPS
    ],
    className="g-2 mb-3",
)

app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                [
                    html.H4(
                        "Autonomous Process Control — Causal ML",
                        className="mt-3 mb-1 text-primary",
                    ),
                    html.P(
                        "Simulates a continuous chemical production process, learns its causal "
                        "structure, detects deviations, and recommends corrective actions.",
                        className="text-muted mb-3",
                    ),
                ]
            )
        ),
        _pipeline_banner,
        dbc.Tabs(
            [
                _process_monitor_tab(),
                _causal_graph_tab(),
                _causal_effects_tab(),
                _live_inference_tab(),
            ],
            active_tab="tab-monitor",
        ),
    ],
    fluid=True,
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@app.callback(
    Output("process-timeseries", "figure"),
    Input("sensor-var-dropdown", "value"),
    Input("show-anomalies-check", "value"),
)
def update_timeseries(selected_vars: list[str], show_anomalies: list[str]) -> go.Figure:
    df = _arts.get("df")
    if df is None or not selected_vars:
        return go.Figure()

    n_rows = len(selected_vars)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=[v.replace("_", " ").title() for v in selected_vars],
    )

    normal_df = df[~df["anomaly_flag"]]
    anomaly_df = df[df["anomaly_flag"]]

    for i, var in enumerate(selected_vars, start=1):
        if var not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=normal_df["batch_id"],
                y=normal_df[var],
                mode="lines",
                name=var,
                legendgroup=var,
                line={"color": "#3498db", "width": 1},
                showlegend=(i == 1),
            ),
            row=i,
            col=1,
        )
        if show_anomalies and len(anomaly_df):
            fig.add_trace(
                go.Scatter(
                    x=anomaly_df["batch_id"],
                    y=anomaly_df[var],
                    mode="markers",
                    name="anomaly",
                    legendgroup="anomaly",
                    marker={"color": "#e74c3c", "size": 4, "symbol": "x"},
                    showlegend=(i == 1),
                ),
                row=i,
                col=1,
            )

    fig.update_layout(
        height=max(300, 160 * n_rows),
        margin={"t": 40, "b": 20},
        legend={"orientation": "h", "y": -0.08},
    )
    return fig


@app.callback(
    Output("inference-output", "children"),
    Input("predict-btn", "n_clicks"),
    [
        State("radio-catalyst", "value"),
        State("slider-reactor_temp", "value"),
        State("slider-coolant_flow_rate", "value"),
        State("slider-pressure", "value"),
        State("slider-ph_level", "value"),
        State("slider-reaction_rate", "value"),
        State("slider-product_yield", "value"),
    ],
    prevent_initial_call=True,
)
def run_inference(
    _n_clicks: int | None,
    catalyst: str,
    reactor_temp: float,
    coolant_flow_rate: float,
    pressure: float,
    ph_level: float,
    reaction_rate: float,
    product_yield: float,
) -> Any:
    detector = _arts.get("detector")
    causal_model = _arts.get("causal_model")
    config = _arts.get("config")

    if detector is None or causal_model is None or config is None:
        return dbc.Alert("Models not loaded. Run 'make train' first.", color="danger")

    reading: dict[str, Any] = {
        "catalyst_type": catalyst,
        "coolant_flow_rate": float(coolant_flow_rate),
        "reactor_temp": float(reactor_temp),
        "pressure": float(pressure),
        "ph_level": float(ph_level),
        "reaction_rate": float(reaction_rate),
        "product_yield": float(product_yield),
    }

    anomaly = detect_anomaly(reading, detector)
    action = recommend_action(anomaly, causal_model, reading, config.control)

    anomaly_color = "danger" if anomaly.flag else "success"
    anomaly_label = "ANOMALY DETECTED" if anomaly.flag else "Normal — within specification"

    action_rows = [
        ("Variable", action.variable),
        ("Current", f"{action.current:.3f}"),
        ("Recommended", f"{action.recommended:.3f}"),
        ("Delta", f"{action.delta:+.3f}"),
        ("Confidence", f"{action.confidence:.1%}"),
    ]

    return [
        dbc.Alert(
            [
                html.Strong(anomaly_label),
                html.Span(
                    f"   type: {anomaly.type}   |   variable: {anomaly.variable}"
                    f"   |   score: {anomaly.score:.4f}"
                ),
            ],
            color=anomaly_color,
            className="mb-3",
        ),
        dbc.Card(
            [
                dbc.CardHeader("Corrective Action"),
                dbc.CardBody(
                    [
                        dbc.Table(
                            [
                                html.Tbody(
                                    [
                                        html.Tr([html.Td(k), html.Td(html.Strong(v))])
                                        for k, v in action_rows
                                    ]
                                )
                            ],
                            bordered=True,
                            size="sm",
                            className="mb-2",
                        ),
                        html.P(action.reasoning, className="text-muted small mb-0"),
                    ]
                ),
            ]
        ),
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    config = load_config()
    logger.info(f"Starting dashboard on http://{config.serving.host}:{config.serving.port}")
    app.run(host=config.serving.host, port=config.serving.port, debug=False)


if __name__ == "__main__":
    main()
