"""Tests for causal_graph.py."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from process_control_causal_ml.causal_graph import (
    _build_dag_from_adjacency,
    _encode_for_discovery,
    _extract_dag_from_causal_learn,
    _get_continuous_columns,
    _map_dag_to_original_names,
    _node_name_to_col,
    compare_to_ground_truth,
    discover_dag,
    get_ground_truth_dag,
    plot_dag,
)
from process_control_causal_ml.utils import CausalGraphConfig


def test_ground_truth_dag_edges() -> None:
    """Ground-truth DAG must have exactly 11 edges."""
    dag = get_ground_truth_dag()
    assert dag.number_of_edges() == 11, f"Expected 11 edges, got {dag.number_of_edges()}"


def test_ground_truth_dag_nodes() -> None:
    """Ground-truth DAG must contain the expected nodes."""
    dag = get_ground_truth_dag()
    expected_nodes = {
        "catalyst_type",
        "coolant_flow_rate",
        "reactor_temp",
        "pressure",
        "ph_level",
        "reaction_rate",
        "product_yield",
    }
    assert expected_nodes == set(dag.nodes())


def test_ground_truth_dag_is_dag() -> None:
    """Ground-truth graph must be a valid DAG (no cycles)."""
    dag = get_ground_truth_dag()
    assert nx.is_directed_acyclic_graph(dag)


def test_discover_dag_returns_digraph(small_df, config) -> None:
    """discover_dag must return a NetworkX DiGraph."""
    dag = discover_dag(small_df, config.causal_graph)
    assert isinstance(dag, nx.DiGraph)


def test_discover_dag_has_expected_nodes(small_df, config) -> None:
    """Discovered DAG must contain process variable nodes."""
    dag = discover_dag(small_df, config.causal_graph)
    # At minimum, the outcome and some upstream variables should appear
    assert dag.number_of_nodes() >= 4


def test_compare_to_ground_truth_returns_shd(small_df, config) -> None:
    """compare_to_ground_truth must return a dict with 'shd' key."""
    learned = discover_dag(small_df, config.causal_graph)
    true_dag = get_ground_truth_dag()
    metrics = compare_to_ground_truth(learned, true_dag)
    assert "shd" in metrics
    assert isinstance(metrics["shd"], int)
    assert metrics["shd"] >= 0


def test_compare_perfect_match() -> None:
    """compare_to_ground_truth should return SHD=0 for identical DAGs."""
    true_dag = get_ground_truth_dag()
    metrics = compare_to_ground_truth(true_dag, true_dag)
    assert metrics["shd"] == 0


def test_compare_empty_dag() -> None:
    """compare_to_ground_truth should return SHD = number of true edges for empty DAG."""
    true_dag = get_ground_truth_dag()
    empty_dag = nx.DiGraph()
    empty_dag.add_nodes_from(true_dag.nodes())
    metrics = compare_to_ground_truth(empty_dag, true_dag)
    assert metrics["missing_edges"] == true_dag.number_of_edges()


def test_plot_dag_creates_file(tmp_path, small_df, config) -> None:
    """plot_dag should create a PNG file."""
    dag = get_ground_truth_dag()
    out = str(tmp_path / "test_dag.png")
    plot_dag(dag, out, title="Test DAG")
    import os

    assert os.path.exists(out)
    assert os.path.getsize(out) > 1000  # non-trivial file


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------


def test_get_continuous_columns_excludes_metadata(small_df) -> None:
    cols = _get_continuous_columns(small_df)
    for excluded in ("batch_id", "timestamp", "anomaly_flag", "anomaly_type", "catalyst_type"):
        assert excluded not in cols
    assert "reactor_temp" in cols


def test_encode_for_discovery_shape(small_df) -> None:
    arr, col_names = _encode_for_discovery(small_df)
    assert isinstance(arr, np.ndarray)
    assert arr.shape[0] == len(small_df)
    assert arr.shape[1] == len(col_names)


def test_encode_for_discovery_drops_metadata(small_df) -> None:
    _, col_names = _encode_for_discovery(small_df)
    for excluded in ("batch_id", "timestamp", "anomaly_flag", "anomaly_type"):
        assert excluded not in col_names
    # catalyst_type itself is dropped; one-hot dummies (minus first) appear instead
    assert "catalyst_type" not in col_names


def test_build_dag_from_adjacency_creates_edges() -> None:
    # adj[i, j] = effect of x_j on x_i  →  edge j → i
    adj = np.zeros((3, 3))
    adj[1, 0] = 0.5  # x0 → x1
    adj[2, 1] = 0.3  # x1 → x2
    cols = ["x0", "x1", "x2"]
    dag = _build_dag_from_adjacency(adj, cols, threshold=0.1)
    assert dag.has_edge("x0", "x1")
    assert dag.has_edge("x1", "x2")
    assert not dag.has_edge("x2", "x0")


def test_build_dag_from_adjacency_threshold() -> None:
    adj = np.zeros((2, 2))
    adj[1, 0] = 0.05  # below default threshold of 0.1
    dag = _build_dag_from_adjacency(adj, ["a", "b"], threshold=0.1)
    assert dag.number_of_edges() == 0


def test_node_name_to_col_valid() -> None:
    cols = ["alpha", "beta", "gamma"]
    assert _node_name_to_col("X1", cols) == "alpha"
    assert _node_name_to_col("X3", cols) == "gamma"


def test_node_name_to_col_out_of_bounds() -> None:
    assert _node_name_to_col("X9", ["a", "b"]) is None


def test_node_name_to_col_invalid_format() -> None:
    assert _node_name_to_col("bad", ["a"]) is None
    assert _node_name_to_col("", ["a"]) is None


def test_map_dag_collapses_onehot_to_catalyst_type() -> None:
    dag = nx.DiGraph()
    dag.add_edge("catalyst_type_B", "reactor_temp")
    dag.add_edge("coolant_flow_rate", "reactor_temp")
    cols = ["coolant_flow_rate", "reactor_temp", "catalyst_type_B"]
    result = _map_dag_to_original_names(dag, cols)
    assert result.has_edge("catalyst_type", "reactor_temp")
    assert result.has_edge("coolant_flow_rate", "reactor_temp")


def test_map_dag_no_self_loops_from_collapsing() -> None:
    dag = nx.DiGraph()
    dag.add_edge("catalyst_type_B", "catalyst_type_C")  # both collapse to catalyst_type
    cols = ["catalyst_type_B", "catalyst_type_C"]
    result = _map_dag_to_original_names(dag, cols)
    assert not any(u == v for u, v in result.edges())


def test_extract_dag_matrix_fallback() -> None:
    """When Endpoint API raises, fall back to reading the .graph matrix."""

    class FakeGraph:
        def get_graph_edges(self):
            raise RuntimeError("no Endpoint API")

        # g[i,j]==-1 and g[j,i]==1  →  edge j → i
        graph = np.array([[0, -1, 0], [1, 0, -1], [0, 1, 0]], dtype=int)

    dag = _extract_dag_from_causal_learn(FakeGraph(), ["a", "b", "c"])
    # g[0,1]==-1, g[1,0]==1  → edge b→a
    # g[1,2]==-1, g[2,1]==1  → edge c→b
    assert dag.has_edge("b", "a")
    assert dag.has_edge("c", "b")


def test_extract_dag_both_fallbacks_fail() -> None:
    """When both paths fail, return an empty DAG with nodes only."""

    class BrokenGraph:
        def get_graph_edges(self):
            raise RuntimeError("Endpoint failed")

        @property
        def graph(self):
            raise RuntimeError("Matrix also failed")

    dag = _extract_dag_from_causal_learn(BrokenGraph(), ["a", "b"])
    assert set(dag.nodes()) == {"a", "b"}
    assert dag.number_of_edges() == 0


def test_compare_extra_edge() -> None:
    true_dag = nx.DiGraph([("a", "b")])
    learned = nx.DiGraph([("a", "b"), ("c", "d")])
    metrics = compare_to_ground_truth(learned, true_dag)
    assert metrics["extra_edges"] >= 1
    assert metrics["true_positive_edges"] == 1


def test_compare_reversed_edge() -> None:
    true_dag = nx.DiGraph([("a", "b")])
    learned = nx.DiGraph([("b", "a")])  # reversed
    metrics = compare_to_ground_truth(learned, true_dag)
    assert metrics["reversed_edges"] == 1
    assert metrics["missing_edges"] == 0


def test_discover_dag_lingam(small_df) -> None:
    cfg = CausalGraphConfig(method="lingam")
    dag = discover_dag(small_df, cfg)
    assert isinstance(dag, nx.DiGraph)
    assert dag.number_of_nodes() >= 1


@pytest.mark.xfail(reason="causal-learn GES has a numpy scalar conversion bug", strict=False)
def test_discover_dag_ges(small_df) -> None:
    cfg = CausalGraphConfig(method="ges")
    dag = discover_dag(small_df, cfg)
    assert isinstance(dag, nx.DiGraph)
    assert dag.number_of_nodes() >= 1


def test_discover_dag_invalid_method_raises(small_df) -> None:
    cfg = CausalGraphConfig.model_construct(method="bad_algo")
    with pytest.raises(ValueError, match="Unknown method"):
        discover_dag(small_df, cfg)
