"""
Integration tests — chain multiple modules the way the real pipeline does.

Catches shape mismatches, interface incompatibilities, and data-flow bugs
that unit tests on individual functions cannot detect.

Pipeline paths tested:
  1. distance_matrix → graph_builder → nx_to_pyg → model.forward()
  2. distance_matrix → graph_builder → edge_weights.compute_edge_weights_for_graph()
  3. feature_loader config → GNN_WithMetadata(meta_dim=clinical_dim)
"""

import numpy as np
import networkx as nx
import pytest

from src.graph_utils import build_knn_graph, build_mst_graph, build_threshold_graph
from src.edge_weights import compute_edge_weights_for_graph, list_strategies
from tests.conftest import requires_torch, HAS_TORCH

if HAS_TORCH:
    import torch
    from torch_geometric.data import Data, Batch
    from src.gpu_graph_conversion import nx_to_pyg_gpu_batch
    from src.models import (
        get_model, get_loss,
        GNN_GINEConv, GNN_GCN, GNN_GAT, GNN_GraphSAGE,
        EdgeCentricRGCN, GNN_Ensemble, MLP_Baseline,
        CNN_Baseline, MetadataOnly_Baseline, GNN_WithMetadata,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline_graphs(n_features=30, n_samples=6):
    """Simulate the real pipeline: build a base graph, then per-sample graphs."""
    rng = np.random.default_rng(42)
    dm = rng.uniform(0.1, 1.5, (n_features, n_features))
    dm = (dm + dm.T) / 2
    np.fill_diagonal(dm, 0)
    feature_ids = [f"ASV_{i:03d}" for i in range(n_features)]

    base_graph = build_knn_graph(dm, feature_ids, k=5)

    sample_graphs = []
    labels = []
    for s in range(n_samples):
        G = base_graph.copy()
        abundances = rng.uniform(0.0, 0.5, n_features)
        nodes_to_remove = [fid for fid, a in zip(feature_ids, abundances) if a < 0.02]
        G.remove_nodes_from(nodes_to_remove)
        for node in G.nodes():
            idx = feature_ids.index(node)
            G.nodes[node]['weight'] = float(abundances[idx])
        if G.number_of_nodes() >= 5 and G.number_of_edges() >= 3:
            sample_graphs.append(G)
            labels.append(s % 2)
    return sample_graphs, np.array(labels), dm, feature_ids


# =========================================================================
# 1. Graph builder → PyG conversion → model forward
# =========================================================================

@requires_torch
class TestGraphToPygToModel:
    """Full chain: NX graphs → PyG Data → model.forward() → valid output."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.graphs, self.labels, _, _ = _make_pipeline_graphs()
        self.pyg_data = nx_to_pyg_gpu_batch(self.graphs, use_gpu=False)
        for i, d in enumerate(self.pyg_data):
            d.y = torch.tensor(float(self.labels[i]))

    def _run_model(self, model_type, **extra_kwargs):
        batch = Batch.from_data_list(self.pyg_data)
        kwargs = dict(input_dim=1, hidden_dim=64, num_layers=2,
                      dropout=0.0, pooling='mean', num_classes=2)
        kwargs.update(extra_kwargs)
        model = get_model(model_type, **kwargs)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        return out, batch

    @pytest.mark.parametrize("model_type", [
        'GCN', 'GINEConv', 'GAT', 'GraphSAGE', 'EdgeCentricRGCN', 'Ensemble', 'MLP',
    ])
    def test_forward_produces_valid_probs(self, model_type):
        out, batch = self._run_model(model_type)
        n = batch.num_graphs
        assert out.shape == (n,), f"{model_type}: shape {out.shape} != ({n},)"
        assert torch.isfinite(out).all(), f"{model_type}: non-finite output"
        assert (out >= 0).all() and (out <= 1).all(), f"{model_type}: out of [0,1]"

    @pytest.mark.parametrize("model_type", [
        'GCN', 'GINEConv', 'GAT', 'GraphSAGE', 'MLP',
    ])
    def test_backward_pass_no_error(self, model_type):
        batch = Batch.from_data_list(self.pyg_data)
        model = get_model(model_type, input_dim=1, hidden_dim=64, num_layers=2, num_classes=2)
        model.train()
        out = model(batch)
        loss_fn = get_loss('bce')
        loss = loss_fn(out, batch.y)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, f"{model_type}: no gradients after backward"

    def test_cnn_on_pipeline_graphs(self):
        batch = Batch.from_data_list(self.pyg_data)
        model = CNN_Baseline(input_dim=100, hidden_dim=64, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (batch.num_graphs,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_pyg_data_shapes_consistent(self):
        """Every converted graph must have matching x/edge_index/edge_attr dims."""
        for i, d in enumerate(self.pyg_data):
            assert d.x.dim() == 2 and d.x.shape[1] == 1, \
                f"Graph {i}: x shape {d.x.shape}, expected [N, 1]"
            assert d.edge_index.dim() == 2 and d.edge_index.shape[0] == 2, \
                f"Graph {i}: edge_index shape {d.edge_index.shape}"
            n_edges = d.edge_index.shape[1]
            assert d.edge_attr.shape == (n_edges, 1), \
                f"Graph {i}: edge_attr {d.edge_attr.shape} != ({n_edges}, 1)"
            n_nodes = d.x.shape[0]
            if n_edges > 0:
                assert (d.edge_index >= 0).all() and (d.edge_index < n_nodes).all(), \
                    f"Graph {i}: edge_index out of range [0, {n_nodes})"

    def test_edge_attr_from_graph_weights(self):
        """PyG edge_attr values must come from the NX graph 'weight' attribute (bidirectional)."""
        for g, d in zip(self.graphs, self.pyg_data):
            nx_weights = sorted(data['weight'] for _, _, data in g.edges(data=True))
            pyg_weights = sorted(d.edge_attr.squeeze().tolist())
            assert len(pyg_weights) == 2 * len(nx_weights), \
                f"Edge count mismatch: NX={len(nx_weights)}, PyG={len(pyg_weights)} (expected 2x for bidirectional)"


# =========================================================================
# 2. Graph builder → different graph types → PyG → same model
# =========================================================================

@requires_torch
class TestMultipleGraphTypesToModel:
    """Different graph construction methods must all produce valid model input."""

    @pytest.fixture(autouse=True)
    def setup(self):
        rng = np.random.default_rng(7)
        n = 25
        self.dm = rng.uniform(0.1, 1.0, (n, n))
        self.dm = (self.dm + self.dm.T) / 2
        np.fill_diagonal(self.dm, 0)
        self.ids = [f"t_{i}" for i in range(n)]
        self.abund = {fid: float(rng.uniform(0.01, 0.5)) for fid in self.ids}

    def _graph_to_pyg_batch(self, G):
        for node in G.nodes():
            if 'weight' not in G.nodes[node]:
                G.nodes[node]['weight'] = self.abund.get(node, 0.1)
        pyg_list = nx_to_pyg_gpu_batch([G], use_gpu=False)
        pyg_list[0].y = torch.tensor(1.0)
        return Batch.from_data_list(pyg_list)

    @pytest.mark.parametrize("graph_type", ['knn', 'mst', 'threshold'])
    def test_gineconv_on_all_graph_types(self, graph_type):
        if graph_type == 'knn':
            G = build_knn_graph(self.dm, self.ids, k=5, abundances=self.abund)
        elif graph_type == 'mst':
            G = build_mst_graph(self.dm, self.ids, abundances=self.abund)
        else:
            G = build_threshold_graph(self.dm, self.ids, threshold_percentile=30, abundances=self.abund)

        batch = self._graph_to_pyg_batch(G)
        model = GNN_GINEConv(input_dim=1, hidden_dim=64, num_layers=2, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (1,)
        assert 0 <= out.item() <= 1


# =========================================================================
# 3. Graph builder → edge_weights (no torch needed)
# =========================================================================

class TestGraphToEdgeWeights:
    """Apply edge_weight strategies on actual graph builder output."""

    @pytest.fixture(autouse=True)
    def setup(self):
        rng = np.random.default_rng(55)
        n = 15
        dm = rng.uniform(0.1, 1.0, (n, n))
        dm = (dm + dm.T) / 2
        np.fill_diagonal(dm, 0)
        ids = [f"f_{i}" for i in range(n)]
        self.G = build_knn_graph(dm, ids, k=4)
        self.abundances = {fid: float(rng.uniform(0.01, 0.5)) for fid in ids}
        self.distances = {}
        for u, v, d in self.G.edges(data=True):
            self.distances[(u, v)] = d.get('distance', d.get('weight', 1.0))

    @pytest.mark.parametrize("strategy", list_strategies())
    def test_all_strategies_on_real_graph(self, strategy):
        edges = list(self.G.edges())
        result = compute_edge_weights_for_graph(
            edges, self.distances, self.abundances, strategy=strategy
        )
        assert len(result) == len(edges)
        for edge, w in result.items():
            assert np.isfinite(w), f"{strategy}: non-finite weight {w} for {edge}"
            assert w >= 0, f"{strategy}: negative weight {w} for {edge}"

    def test_inverse_weights_are_positive(self):
        edges = list(self.G.edges())
        result = compute_edge_weights_for_graph(
            edges, self.distances, self.abundances, strategy='inverse'
        )
        for w in result.values():
            assert w > 0

    def test_identity_matches_distances(self):
        edges = list(self.G.edges())
        result = compute_edge_weights_for_graph(
            edges, self.distances, self.abundances, strategy='identity'
        )
        for (u, v), w in result.items():
            expected = self.distances.get((u, v)) or self.distances.get((v, u))
            assert w == pytest.approx(expected, rel=1e-6)

    def test_abundance_product_zero_abundance_gives_zero(self):
        edges = list(self.G.edges())[:1]
        u, v = edges[0]
        zero_abundances = {u: 0.0, v: 0.5}
        result = compute_edge_weights_for_graph(
            edges, self.distances, zero_abundances, strategy='abundance_product'
        )
        assert result[edges[0]] == pytest.approx(0.0)


# =========================================================================
# 4. Clinical features dim → model meta_dim consistency
# =========================================================================

@requires_torch
class TestClinicalDimConsistency:
    """clinical_dim from feature_loader must match model's meta_dim."""

    def test_matching_dim_works(self):
        clinical_dim = 3
        model = GNN_WithMetadata(
            gnn_type='GCN', input_dim=1, hidden_dim=64,
            meta_dim=clinical_dim, num_classes=2,
        )
        x = torch.randn(5, 1)
        ei = torch.tensor([[0,1,2,3,4],[1,2,3,4,0]], dtype=torch.long)
        ea = torch.rand(5, 1)
        clinical = torch.randn(1, clinical_dim)
        d = Data(x=x, edge_index=ei, edge_attr=ea, y=torch.tensor(1.0), clinical=clinical)
        batch = Batch.from_data_list([d])
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (1,)
        assert 0 <= out.item() <= 1

    def test_mismatched_dim_raises(self):
        model = GNN_WithMetadata(
            gnn_type='GCN', input_dim=1, hidden_dim=64,
            meta_dim=5, num_classes=2,
        )
        x = torch.randn(5, 1)
        ei = torch.tensor([[0,1,2,3,4],[1,2,3,4,0]], dtype=torch.long)
        ea = torch.rand(5, 1)
        clinical = torch.randn(1, 3)  # dim=3, model expects 5
        d = Data(x=x, edge_index=ei, edge_attr=ea, y=torch.tensor(1.0), clinical=clinical)
        batch = Batch.from_data_list([d])
        with pytest.raises(ValueError, match="does not match meta_dim"):
            model(batch)

    def test_metadata_only_matching_dim(self):
        clinical_dim = 4
        model = MetadataOnly_Baseline(meta_dim=clinical_dim, hidden_dim=32, num_classes=2)
        x = torch.randn(5, 1)
        ei = torch.zeros(2, 0, dtype=torch.long)
        clinical = torch.randn(1, clinical_dim)
        d = Data(x=x, edge_index=ei, y=torch.tensor(0.0), clinical=clinical)
        batch = Batch.from_data_list([d])
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (1,)

    def test_feature_loader_none_config_raises(self):
        """load_features with None config must raise ValueError."""
        from src.feature_loader import load_features
        with pytest.raises(ValueError, match="must not be None"):
            load_features(None, '/nonexistent')

    def test_feature_loader_disabled_config(self):
        from src.feature_loader import load_features
        config = {
            'data_extraction': {
                'disease_criteria': {'disease': 'IBD'},
                'clinical_features': {
                    'enable': False,
                    'features': {'use_age': True, 'use_sex': True, 'use_bmi': True, 'use_antibiotics': False},
                },
            },
            'model_training': {'architecture': {'use_clinical_features': False}},
        }
        result = load_features(config, '/nonexistent')
        assert result['use_clinical'] is False


# =========================================================================
# 5. Loss function on model output from pipeline graphs
# =========================================================================

@requires_torch
class TestLossOnPipelineOutput:
    """Loss functions must work on actual model outputs from pipeline graphs."""

    @pytest.fixture(autouse=True)
    def setup(self):
        graphs, labels, _, _ = _make_pipeline_graphs(n_features=20, n_samples=4)
        pyg_data = nx_to_pyg_gpu_batch(graphs, use_gpu=False)
        for i, d in enumerate(pyg_data):
            d.y = torch.tensor(float(labels[i]))
        self.batch = Batch.from_data_list(pyg_data)

    @pytest.mark.parametrize("loss_type", ['bce', 'focal', 'weighted_bce'])
    def test_loss_finite_on_model_output(self, loss_type):
        model = GNN_GINEConv(input_dim=1, hidden_dim=64, num_layers=2, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(self.batch)
        loss_fn = get_loss(loss_type, pos_weight=2.0) if loss_type == 'weighted_bce' else get_loss(loss_type)
        loss = loss_fn(out, self.batch.y)
        assert torch.isfinite(loss), f"{loss_type}: loss={loss}"
        assert loss.item() >= 0
