"""Causal DAG discovery from observational process data."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from process_control_causal_ml.utils import CausalGraphConfig, load_config, logger

DATA_DIR = Path("data")

# ---------------------------------------------------------------------------
# Ground-truth DAG
# ---------------------------------------------------------------------------


def get_ground_truth_dag() -> nx.DiGraph:
    """Return the ground-truth causal DAG as a NetworkX DiGraph."""
    dag = nx.DiGraph()
    edges = [
        ("catalyst_type", "reactor_temp"),
        ("coolant_flow_rate", "reactor_temp"),
        ("coolant_flow_rate", "ph_level"),
        ("reactor_temp", "pressure"),
        ("reactor_temp", "reaction_rate"),
        ("reactor_temp", "product_yield"),
        ("pressure", "reaction_rate"),
        ("pressure", "ph_level"),
        ("ph_level", "reaction_rate"),
        ("ph_level", "product_yield"),
        ("reaction_rate", "product_yield"),
    ]
    dag.add_edges_from(edges)
    return dag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_continuous_columns(df: pd.DataFrame) -> list[str]:
    """Return continuous process variable columns (exclude metadata)."""
    exclude = {"batch_id", "timestamp", "anomaly_flag", "anomaly_type", "catalyst_type"}
    return [c for c in df.columns if c not in exclude]


def _encode_for_discovery(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """One-hot encode categoricals and return numpy array + column names."""
    exclude = {"batch_id", "timestamp", "anomaly_flag", "anomaly_type"}
    data = df.drop(columns=[c for c in exclude if c in df.columns])
    data = pd.get_dummies(data, columns=["catalyst_type"], drop_first=True)
    col_names = list(data.columns)
    return data.values.astype(float), col_names


def _build_dag_from_adjacency(
    adj: np.ndarray, col_names: list[str], threshold: float = 0.1
) -> nx.DiGraph:
    """Build DiGraph from LiNGAM adjacency matrix.
    adj[i, j] = causal effect of x_j on x_i (edge: j -> i).
    """
    dag = nx.DiGraph()
    dag.add_nodes_from(col_names)
    n = len(col_names)
    for i in range(n):
        for j in range(n):
            if abs(adj[i, j]) > threshold:
                dag.add_edge(col_names[j], col_names[i])
    return dag


def _node_name_to_col(name: str, col_names: list[str]) -> str | None:
    """Convert a causal-learn node name ('X1', 'X2', …) to a column name. Returns None on failure."""
    try:
        idx = int(name[1:]) - 1
    except (ValueError, IndexError):
        return None
    return col_names[idx] if idx < len(col_names) else None


def _add_edges_from_endpoints(cg_graph: Any, col_names: list[str], dag: nx.DiGraph) -> None:
    """Populate dag with directed edges using the causal-learn Endpoint API."""
    from causallearn.graph.Endpoint import Endpoint

    for edge in cg_graph.get_graph_edges():
        col1 = _node_name_to_col(edge.get_node1().get_name(), col_names)
        col2 = _node_name_to_col(edge.get_node2().get_name(), col_names)
        if col1 is None or col2 is None:
            continue
        ep1, ep2 = edge.get_endpoint1(), edge.get_endpoint2()
        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            dag.add_edge(col1, col2)
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.TAIL:
            dag.add_edge(col2, col1)
        # Undirected edges: skip — can't orient without additional assumptions


def _add_edges_from_matrix(cg_graph: Any, col_names: list[str], dag: nx.DiGraph) -> None:
    """Populate dag from the raw adjacency matrix (fallback when Endpoint API unavailable).

    Convention: g[i, j] == -1 and g[j, i] == 1  →  edge j → i.
    """
    g = cg_graph.graph
    n = min(g.shape[0], len(col_names))
    for i in range(n):
        for j in range(i + 1, n):
            if g[i, j] == -1 and g[j, i] == 1:
                dag.add_edge(col_names[j], col_names[i])
            elif g[i, j] == 1 and g[j, i] == -1:
                dag.add_edge(col_names[i], col_names[j])


def _extract_dag_from_causal_learn(cg_graph: Any, col_names: list[str]) -> nx.DiGraph:
    """Extract directed edges from a causal-learn GeneralGraph (CPDAG output of PC/GES)."""
    dag = nx.DiGraph()
    dag.add_nodes_from(col_names)
    try:
        _add_edges_from_endpoints(cg_graph, col_names, dag)
    except Exception as exc:
        logger.warning(f"Endpoint API failed ({exc}); falling back to graph matrix")
        try:
            _add_edges_from_matrix(cg_graph, col_names, dag)
        except Exception as exc2:
            logger.error(f"Matrix fallback also failed: {exc2}")
    return dag


def _map_dag_to_original_names(dag: nx.DiGraph, col_names: list[str]) -> nx.DiGraph:
    """Collapse one-hot encoded catalyst_type columns back to 'catalyst_type'."""
    mapping: dict[str, str] = {}
    for col in col_names:
        if col.startswith("catalyst_type_"):
            mapping[col] = "catalyst_type"
        else:
            mapping[col] = col

    new_dag = nx.DiGraph()
    for u, v in dag.edges():
        new_u = mapping.get(u, u)
        new_v = mapping.get(v, v)
        if new_u != new_v:  # avoid self-loops from collapsing
            new_dag.add_edge(new_u, new_v)

    return new_dag


# ---------------------------------------------------------------------------
# Discovery algorithms
# ---------------------------------------------------------------------------


def _discover_pc(data: np.ndarray, col_names: list[str], config: CausalGraphConfig) -> nx.DiGraph:
    from causallearn.search.ConstraintBased.PC import pc

    logger.info("Running PC algorithm...")
    cg = pc(
        data,
        alpha=config.significance_level,
        indep_test="fisherz",
        stable=True,
        uc_rule=0,
        uc_priority=-1,
        show_progress=False,
    )
    dag = _extract_dag_from_causal_learn(cg.G, col_names)
    return dag


def _discover_lingam(
    data: np.ndarray, col_names: list[str], config: CausalGraphConfig
) -> nx.DiGraph:
    import lingam

    logger.info("Running DirectLiNGAM...")
    model = lingam.DirectLiNGAM(random_state=42)
    model.fit(data)
    dag = _build_dag_from_adjacency(model.adjacency_matrix_, col_names, threshold=0.05)
    return dag


def _discover_ges(data: np.ndarray, col_names: list[str], config: CausalGraphConfig) -> nx.DiGraph:
    from causallearn.search.ScoreBased.GES import ges

    logger.info("Running GES...")
    record = ges(data, score_func="local_score_BIC", maxP=None, parameters=None)
    dag = _extract_dag_from_causal_learn(record["G"], col_names)
    return dag


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def discover_dag(df: pd.DataFrame, config: CausalGraphConfig) -> nx.DiGraph:
    """Discover causal DAG from observational data.

    Subsamples to 5,000 rows for performance. Uses method from config.
    One-hot encodes catalyst_type before discovery, then maps back.
    """
    # Use only normal data for graph discovery
    normal = df[~df["anomaly_flag"]].copy() if "anomaly_flag" in df.columns else df.copy()

    # Subsample to 5000 rows
    n_sample = min(5000, len(normal))
    sample = normal.sample(n=n_sample, random_state=42)

    data_arr, col_names = _encode_for_discovery(sample)

    logger.info(
        f"Discovering DAG via '{config.method}' on {n_sample} samples, {len(col_names)} variables"
    )

    method = config.method.lower()
    if method == "pc":
        dag_encoded = _discover_pc(data_arr, col_names, config)
    elif method == "lingam":
        dag_encoded = _discover_lingam(data_arr, col_names, config)
    elif method == "ges":
        dag_encoded = _discover_ges(data_arr, col_names, config)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Map one-hot columns back to original variable names
    dag = _map_dag_to_original_names(dag_encoded, col_names)
    logger.info(f"Discovered DAG: {dag.number_of_nodes()} nodes, {dag.number_of_edges()} edges")
    return dag


def compare_to_ground_truth(learned: nx.DiGraph, true_dag: nx.DiGraph) -> dict[str, int | float]:
    """Compute Structural Hamming Distance (SHD) and related metrics.

    SHD = missing edges + extra edges + reversed edges.
    """
    true_edges = set(true_dag.edges())
    learned_edges = set(learned.edges())
    reversed_learned = {(v, u) for u, v in learned_edges}
    reversed_true = {(v, u) for u, v in true_edges}

    missing = true_edges - learned_edges - reversed_learned
    extra = learned_edges - true_edges - reversed_true
    reversed_e = true_edges & reversed_learned

    shd = len(missing) + len(extra) + len(reversed_e)
    tp = len(true_edges & learned_edges)

    results = {
        "shd": shd,
        "missing_edges": len(missing),
        "extra_edges": len(extra),
        "reversed_edges": len(reversed_e),
        "true_positive_edges": tp,
        "precision": tp / max(len(learned_edges), 1),
        "recall": tp / max(len(true_edges), 1),
    }
    logger.info(
        f"DAG comparison — SHD: {shd}, precision: {results['precision']:.2f}, recall: {results['recall']:.2f}"
    )
    return results


def plot_dag(dag: nx.DiGraph, path: str, title: str = "Learned Causal DAG") -> None:
    """Save a DAG plot as a PNG file."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Hierarchical layout
    try:
        pos = nx.nx_agraph.graphviz_layout(dag, prog="dot")
    except Exception:
        pos = nx.spring_layout(dag, seed=42, k=2)

    node_colors = []
    for node in dag.nodes():
        if node == "product_yield":
            node_colors.append("#e74c3c")
        elif node in ("catalyst_type", "coolant_flow_rate"):
            node_colors.append("#2ecc71")
        else:
            node_colors.append("#3498db")

    nx.draw_networkx(
        dag,
        pos=pos,
        ax=ax,
        node_color=node_colors,
        node_size=2000,
        font_size=9,
        font_weight="bold",
        arrows=True,
        arrowsize=20,
        edge_color="#555555",
        width=1.5,
    )
    ax.set_title(title, fontsize=13, pad=15)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved DAG plot to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(evaluate: bool = False) -> None:
    config = load_config()
    DATA_DIR.mkdir(exist_ok=True)

    data_path = DATA_DIR / "process_data.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found. Run 'make simulate' first.")

    df = pd.read_parquet(data_path)
    logger.info(f"Loaded data: {df.shape}")

    dag = discover_dag(df, config.causal_graph)

    # Save
    joblib.dump(dag, DATA_DIR / "causal_graph.pkl")

    plot_dag(
        dag,
        str(DATA_DIR / "causal_graph.png"),
        title=f"Learned DAG ({config.causal_graph.method.upper()})",
    )

    if evaluate:
        true_dag = get_ground_truth_dag()
        metrics = compare_to_ground_truth(dag, true_dag)
        import json

        with open(DATA_DIR / "dag_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"DAG metrics: {metrics}")

        # Also plot ground truth
        plot_dag(true_dag, str(DATA_DIR / "causal_graph_true.png"), title="Ground-Truth Causal DAG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    main(evaluate=args.evaluate)
