"""Tests for src/gpu_graph_conversion.py — NX→PyG, edge directions, empty graphs, string nodes."""

import pytest
import numpy as np
import networkx as nx
from tests.conftest import requires_torch, HAS_TORCH

if HAS_TORCH:
    import torch
    from src.gpu_graph_conversion import nx_to_pyg_gpu_batch, _convert_batch_parallel


@requires_torch
class TestNxToPygConversion:
    def _make_graph(self, n_nodes=5, n_edges=6):
        G = nx.gnm_random_graph(n_nodes, n_edges, seed=42)
        for node in G.nodes():
            G.nodes[node]['weight'] = float(np.random.rand())
        for u, v in G.edges():
            G[u][v]['weight'] = float(np.random.rand())
        return G

    def test_basic_conversion(self):
        G = self._make_graph()
        results = nx_to_pyg_gpu_batch([G], use_gpu=False)
        assert len(results) == 1
        data = results[0]
        assert data.x.shape[0] == G.number_of_nodes()
        assert data.x.shape[1] == 1
        assert data.edge_index.shape[0] == 2

    def test_empty_graph_placeholder(self):
        G = nx.Graph()
        results = nx_to_pyg_gpu_batch([G], use_gpu=False)
        data = results[0]
        assert data.x.shape == (1, 1)
        assert data.edge_index.shape == (2, 0)
        assert data.edge_attr.shape == (0, 1)

    def test_graph_with_nodes_no_edges(self):
        G = nx.Graph()
        G.add_node(0, weight=0.5)
        G.add_node(1, weight=0.3)
        results = nx_to_pyg_gpu_batch([G], use_gpu=False)
        data = results[0]
        assert data.x.shape[0] == 2
        assert data.edge_index.shape[1] == 0

    def test_batch_multiple_graphs(self):
        graphs = [self._make_graph(n_nodes=i+3, n_edges=i+2) for i in range(5)]
        results = nx_to_pyg_gpu_batch(graphs, use_gpu=False, batch_size=2)
        assert len(results) == 5

    def test_edge_attr_shape(self):
        G = self._make_graph()
        results = nx_to_pyg_gpu_batch([G], use_gpu=False)
        data = results[0]
        assert data.edge_attr is not None
        assert data.edge_attr.shape[1] == 1
        assert data.edge_attr.shape[0] == data.edge_index.shape[1]

    def test_data_on_cpu(self):
        G = self._make_graph()
        results = nx_to_pyg_gpu_batch([G], use_gpu=False)
        data = results[0]
        assert data.x.device.type == 'cpu'
        assert data.edge_index.device.type == 'cpu'
        assert data.edge_attr.device.type == 'cpu'

    def test_node_weight_preserved(self):
        G = nx.Graph()
        G.add_node('a', weight=0.42)
        G.add_node('b', weight=0.99)
        G.add_edge('a', 'b', weight=1.0)
        results = nx_to_pyg_gpu_batch([G], use_gpu=False)
        data = results[0]
        weights = sorted(data.x.squeeze().tolist())
        assert weights == pytest.approx([0.42, 0.99])

    def test_edge_weight_preserved(self):
        G = nx.Graph()
        G.add_node(0, weight=0.1)
        G.add_node(1, weight=0.2)
        G.add_edge(0, 1, weight=0.777)
        results = nx_to_pyg_gpu_batch([G], use_gpu=False)
        data = results[0]
        weights = data.edge_attr.squeeze().tolist()
        assert any(abs(w - 0.777) < 1e-3 for w in weights)

    def test_string_node_names(self):
        """Real pipeline uses ASV sequence strings as node names."""
        G = nx.Graph()
        G.add_node("TACGGA", weight=0.1)
        G.add_node("ATCGCC", weight=0.2)
        G.add_edge("TACGGA", "ATCGCC", weight=0.5)
        results = nx_to_pyg_gpu_batch([G], use_gpu=False)
        data = results[0]
        assert data.x.shape[0] == 2
        assert data.edge_index.shape[1] >= 1

    def test_missing_node_weight_defaults_to_zero(self):
        G = nx.Graph()
        G.add_node(0)  # no weight attribute
        G.add_node(1)
        G.add_edge(0, 1, weight=1.0)
        results = nx_to_pyg_gpu_batch([G], use_gpu=False)
        data = results[0]
        assert data.x[0, 0].item() == 0.0

    def test_missing_edge_weight_defaults_to_one(self):
        G = nx.Graph()
        G.add_node(0, weight=0.1)
        G.add_node(1, weight=0.2)
        G.add_edge(0, 1)  # no weight attribute
        results = nx_to_pyg_gpu_batch([G], use_gpu=False)
        data = results[0]
        assert data.edge_attr[0, 0].item() == pytest.approx(1.0)

    def test_large_batch(self):
        graphs = [self._make_graph(n_nodes=10, n_edges=15) for _ in range(100)]
        results = nx_to_pyg_gpu_batch(graphs, use_gpu=False, batch_size=32)
        assert len(results) == 100
        for data in results:
            assert data.x.shape[0] == 10

    def test_edge_index_valid_range(self):
        """edge_index values must be in [0, n_nodes)."""
        G = self._make_graph(n_nodes=8, n_edges=12)
        results = nx_to_pyg_gpu_batch([G], use_gpu=False)
        data = results[0]
        n = data.x.shape[0]
        assert (data.edge_index >= 0).all()
        assert (data.edge_index < n).all()
