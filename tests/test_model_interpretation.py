"""Tests for src/model_interpretation.py — gradients, importance, aggregation."""

import pytest
import numpy as np
from tests.conftest import requires_torch, HAS_TORCH

if HAS_TORCH:
    import torch
    from torch_geometric.data import Data, Batch
    from src.models import get_model
    from src.model_interpretation import (
        compute_node_gradients,
        compute_integrated_gradients,
        get_top_important_nodes,
        aggregate_importance_across_samples,
        analyze_prediction,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pyg_data(n_nodes=10, device="cpu"):
    """Create a minimal PyG Data for interpretation tests."""
    x = torch.randn(n_nodes, 1)
    edge_index = torch.tensor(
        [[i, (i + 1) % n_nodes] for i in range(n_nodes)]
        + [[(i + 1) % n_nodes, i] for i in range(n_nodes)],
        dtype=torch.long,
    ).t()
    edge_attr = torch.ones(edge_index.shape[1], 1)
    batch = torch.zeros(n_nodes, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                batch=batch, y=torch.tensor(1.0)).to(device)


# ---------------------------------------------------------------------------
# Tests: compute_node_gradients
# ---------------------------------------------------------------------------

@requires_torch
class TestComputeNodeGradients:
    def test_output_shape(self):
        model = get_model("GCN", input_dim=1, hidden_dim=16, num_layers=2, num_classes=2)
        data = _make_pyg_data(n_nodes=10)
        importance = compute_node_gradients(model, data, device="cpu")
        assert importance.shape == (10,)

    def test_output_finite(self):
        model = get_model("GCN", input_dim=1, hidden_dim=16, num_layers=2, num_classes=2)
        data = _make_pyg_data(n_nodes=10)
        importance = compute_node_gradients(model, data, device="cpu")
        assert np.all(np.isfinite(importance))

    def test_output_non_negative(self):
        model = get_model("GCN", input_dim=1, hidden_dim=16, num_layers=2, num_classes=2)
        data = _make_pyg_data(n_nodes=10)
        importance = compute_node_gradients(model, data, device="cpu")
        assert np.all(importance >= 0)

    @pytest.mark.parametrize("model_type", ["GCN", "GINEConv", "GAT", "MLP"])
    def test_works_for_multiple_models(self, model_type):
        model = get_model(model_type, input_dim=1, hidden_dim=16, num_layers=2, num_classes=2)
        data = _make_pyg_data(n_nodes=10)
        importance = compute_node_gradients(model, data, device="cpu")
        assert importance.shape == (10,)
        assert np.all(np.isfinite(importance))


# ---------------------------------------------------------------------------
# Tests: compute_integrated_gradients
# ---------------------------------------------------------------------------

@requires_torch
class TestComputeIntegratedGradients:
    def test_output_shape(self):
        model = get_model("GCN", input_dim=1, hidden_dim=16, num_layers=2, num_classes=2)
        data = _make_pyg_data(n_nodes=10)
        importance = compute_integrated_gradients(model, data, steps=5, device="cpu")
        assert importance.shape == (10,)

    def test_output_finite(self):
        model = get_model("GCN", input_dim=1, hidden_dim=16, num_layers=2, num_classes=2)
        data = _make_pyg_data(n_nodes=10)
        importance = compute_integrated_gradients(model, data, steps=5, device="cpu")
        assert np.all(np.isfinite(importance))

    def test_custom_baseline(self):
        model = get_model("GCN", input_dim=1, hidden_dim=16, num_layers=2, num_classes=2)
        data = _make_pyg_data(n_nodes=10)
        baseline = torch.ones(10, 1) * 0.5
        importance = compute_integrated_gradients(
            model, data, steps=5, baseline=baseline, device="cpu"
        )
        assert importance.shape == (10,)
        assert np.all(np.isfinite(importance))


# ---------------------------------------------------------------------------
# Tests: get_top_important_nodes
# ---------------------------------------------------------------------------

class TestGetTopImportantNodes:
    def test_returns_correct_count(self):
        importance = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        top = get_top_important_nodes(importance, top_k=3)
        assert len(top) == 3

    def test_sorted_descending(self):
        importance = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        top = get_top_important_nodes(importance, top_k=5)
        scores = [s for _, s in top]
        assert scores == sorted(scores, reverse=True)

    def test_with_node_ids(self):
        importance = np.array([0.1, 0.9, 0.5])
        node_ids = ["A", "B", "C"]
        top = get_top_important_nodes(importance, node_ids=node_ids, top_k=2)
        assert top[0][0] == "B"
        assert top[0][1] == pytest.approx(0.9)

    def test_top_k_exceeds_length(self):
        importance = np.array([0.1, 0.9])
        top = get_top_important_nodes(importance, top_k=10)
        assert len(top) == 2


# ---------------------------------------------------------------------------
# Tests: aggregate_importance_across_samples
# ---------------------------------------------------------------------------

class TestAggregateImportance:
    def test_mean_aggregation(self):
        imp_list = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])]
        result = aggregate_importance_across_samples(imp_list, method="mean")
        assert result.shape == (3,)
        np.testing.assert_allclose(result, [2.0, 2.0, 2.0])

    def test_max_aggregation(self):
        imp_list = [np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])]
        result = aggregate_importance_across_samples(imp_list, method="max")
        np.testing.assert_allclose(result, [3.0, 2.0, 3.0])

    def test_median_aggregation(self):
        imp_list = [np.array([1.0, 3.0, 5.0]), np.array([2.0, 3.0, 4.0])]
        result = aggregate_importance_across_samples(imp_list, method="median")
        np.testing.assert_allclose(result, [1.5, 3.0, 4.5])

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            aggregate_importance_across_samples([np.array([1.0])], method="invalid")

    def test_different_lengths_padded(self):
        imp_list = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])]
        result = aggregate_importance_across_samples(imp_list, method="mean")
        assert result.shape == (3,)
        assert result[2] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Tests: analyze_prediction
# ---------------------------------------------------------------------------

@requires_torch
class TestAnalyzePrediction:
    def test_returns_expected_keys(self):
        model = get_model("GCN", input_dim=1, hidden_dim=16, num_layers=2, num_classes=2)
        data = _make_pyg_data(n_nodes=10)
        result = analyze_prediction(model, data, device="cpu", method="gradient")
        assert "predicted_class" in result
        assert "confidence" in result
        assert "node_importance" in result
        assert "top_nodes" in result
        assert "method" in result
        assert "num_nodes" in result

    def test_gradient_method(self):
        model = get_model("GCN", input_dim=1, hidden_dim=16, num_layers=2, num_classes=2)
        data = _make_pyg_data(n_nodes=10)
        result = analyze_prediction(model, data, device="cpu", method="gradient")
        assert result["method"] == "gradient"
        assert result["num_nodes"] == 10
        assert result["predicted_class"] in (0, 1)

    def test_integrated_gradients_method(self):
        model = get_model("GCN", input_dim=1, hidden_dim=16, num_layers=2, num_classes=2)
        data = _make_pyg_data(n_nodes=10)
        result = analyze_prediction(model, data, device="cpu", method="integrated_gradients")
        assert result["method"] == "integrated_gradients"

    def test_unknown_method_raises(self):
        model = get_model("GCN", input_dim=1, hidden_dim=16, num_layers=2, num_classes=2)
        data = _make_pyg_data(n_nodes=10)
        with pytest.raises(ValueError, match="Unknown method"):
            analyze_prediction(model, data, device="cpu", method="invalid")
