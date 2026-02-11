"""Tests for src/models.py — forward shapes, probability ranges, gradients, factory."""

import pytest
import numpy as np
from tests.conftest import requires_torch, HAS_TORCH

if HAS_TORCH:
    import torch
    from torch_geometric.data import Data, Batch
    from src.models import (
        get_model, get_loss,
        FocalLoss, BCELossClamped, WeightedBCELoss,
        MLP_Baseline, GNN_GCN, GNN_GINEConv, GNN_GAT,
        GNN_GraphSAGE, GNN_Ensemble, EdgeCentricRGCN,
        CNN_Baseline, MetadataOnly_Baseline, GNN_WithMetadata,
        GNN_Clinical, SklearnModelWrapper,
    )

BINARY_GNN_TYPES = ['GCN', 'GINEConv', 'GAT', 'GraphSAGE', 'EdgeCentricRGCN', 'Ensemble', 'MLP']
ALL_POOLINGS = ['mean', 'max', 'sum']


# =========================================================================
# Model forward — shape and probability range
# =========================================================================

@requires_torch
class TestBinaryGNNForward:
    @pytest.mark.parametrize("model_type", BINARY_GNN_TYPES)
    def test_output_shape_and_range(self, pyg_batch, model_type):
        kwargs = dict(input_dim=1, hidden_dim=64, num_layers=2, dropout=0.0, pooling='mean', num_classes=2)
        model = get_model(model_type, **kwargs)
        model.eval()
        with torch.no_grad():
            out = model(pyg_batch)
        assert out.shape == (2,), f"{model_type}: expected shape (2,), got {out.shape}"
        assert torch.isfinite(out).all(), f"{model_type}: output has NaN/Inf"
        assert (out >= 0).all() and (out <= 1).all(), f"{model_type}: output outside [0,1]: {out}"

    @pytest.mark.parametrize("model_type", BINARY_GNN_TYPES)
    def test_forward_embedding_shape(self, pyg_batch, model_type):
        kwargs = dict(input_dim=1, hidden_dim=64, num_layers=2, dropout=0.0, pooling='mean', num_classes=2)
        model = get_model(model_type, **kwargs)
        model.eval()
        with torch.no_grad():
            emb = model.forward_embedding(pyg_batch)
        assert emb.dim() == 2
        assert emb.shape[0] == 2

    @pytest.mark.parametrize("model_type", BINARY_GNN_TYPES)
    def test_gradient_flows(self, pyg_batch, model_type):
        """Backward pass must not error and gradients must exist."""
        hidden = 64 if model_type != 'Ensemble' else 128
        kwargs = dict(input_dim=1, hidden_dim=hidden, num_layers=2, dropout=0.0, pooling='mean', num_classes=2)
        model = get_model(model_type, **kwargs)
        model.train()
        out = model(pyg_batch)
        target = torch.ones_like(out)
        loss = torch.nn.functional.binary_cross_entropy(out.clamp(1e-6, 1-1e-6), target)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
        assert has_grad, f"{model_type}: no gradient flow detected"

    @pytest.mark.parametrize("model_type", ['GCN', 'GINEConv', 'MLP'])
    def test_single_graph_batch(self, model_type):
        """Batch size 1 must not crash."""
        x = torch.randn(5, 1)
        ei = torch.tensor([[0,1,2,3,4],[1,2,3,4,0]], dtype=torch.long)
        ea = torch.rand(5, 1)
        d = Data(x=x, edge_index=ei, edge_attr=ea, y=torch.tensor(1.0))
        batch = Batch.from_data_list([d])
        model = get_model(model_type, input_dim=1, hidden_dim=64, num_layers=2, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (1,)


# =========================================================================
# Pooling modes
# =========================================================================

@requires_torch
class TestPoolingModes:
    @pytest.mark.parametrize("pooling", ALL_POOLINGS)
    def test_gcn_all_poolings(self, pyg_batch, pooling):
        model = GNN_GCN(input_dim=1, hidden_dim=64, num_layers=2, pooling=pooling, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(pyg_batch)
        assert out.shape == (2,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_invalid_pooling_raises(self):
        with pytest.raises(ValueError, match="pooling must be"):
            GNN_GCN(input_dim=1, hidden_dim=64, pooling='invalid')


# =========================================================================
# CNN Baseline
# =========================================================================

@requires_torch
class TestCNNBaseline:
    def test_forward_shape(self):
        x = torch.randn(8, 1)
        batch = Batch.from_data_list([
            Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long), y=torch.tensor(0.0)),
            Data(x=x.clone(), edge_index=torch.zeros(2, 0, dtype=torch.long), y=torch.tensor(1.0)),
        ])
        model = CNN_Baseline(input_dim=8, hidden_dim=64, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (2,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_minimal_graph_size(self):
        """Must handle graph with only 4 nodes (min for MaxPool1d(2) x2)."""
        d = Data(x=torch.randn(4, 1), edge_index=torch.zeros(2, 0, dtype=torch.long), y=torch.tensor(0.0))
        batch = Batch.from_data_list([d])
        model = CNN_Baseline(input_dim=4, hidden_dim=64, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (1,)
        assert torch.isfinite(out).all()

    def test_embedding_shape(self):
        d = Data(x=torch.randn(10, 1), edge_index=torch.zeros(2, 0, dtype=torch.long))
        batch = Batch.from_data_list([d])
        model = CNN_Baseline(input_dim=10, hidden_dim=64, num_classes=2)
        model.eval()
        with torch.no_grad():
            emb = model.forward_embedding(batch)
        assert emb.dim() == 2 and emb.shape[0] == 1


# =========================================================================
# Metadata models
# =========================================================================

@requires_torch
class TestMetadataModels:
    def test_metadata_only_forward(self, pyg_batch_with_clinical):
        model = MetadataOnly_Baseline(meta_dim=3, hidden_dim=32, num_layers=2, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(pyg_batch_with_clinical)
        assert out.shape == (2,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_metadata_only_missing_clinical_raises(self, pyg_batch):
        model = MetadataOnly_Baseline(meta_dim=3, hidden_dim=32, num_classes=2)
        with pytest.raises(ValueError, match="requires data.clinical"):
            model(pyg_batch)

    def test_metadata_only_wrong_dim_raises(self, pyg_batch_with_clinical):
        model = MetadataOnly_Baseline(meta_dim=99, hidden_dim=32, num_classes=2)
        with pytest.raises(ValueError, match="does not match meta_dim"):
            model(pyg_batch_with_clinical)

    def test_metadata_only_embedding(self, pyg_batch_with_clinical):
        model = MetadataOnly_Baseline(meta_dim=3, hidden_dim=32, num_layers=2, num_classes=2)
        model.eval()
        with torch.no_grad():
            emb = model.forward_embedding(pyg_batch_with_clinical)
        assert emb.dim() == 2 and emb.shape[0] == 2

    @pytest.mark.parametrize("gnn_type", ['GCN', 'GINEConv', 'GAT', 'GraphSAGE'])
    def test_gnn_with_metadata_all_gnn_types(self, pyg_batch_with_clinical, gnn_type):
        model = GNN_WithMetadata(gnn_type=gnn_type, input_dim=1, hidden_dim=64, meta_dim=3, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(pyg_batch_with_clinical)
        assert out.shape == (2,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_gnn_with_metadata_wrong_dim_raises(self, pyg_batch_with_clinical):
        model = GNN_WithMetadata(gnn_type='GCN', input_dim=1, hidden_dim=64, meta_dim=99, num_classes=2)
        with pytest.raises(ValueError, match="does not match meta_dim"):
            model(pyg_batch_with_clinical)

    def test_gnn_with_metadata_missing_clinical_raises(self, pyg_batch):
        model = GNN_WithMetadata(gnn_type='GCN', input_dim=1, hidden_dim=64, meta_dim=3, num_classes=2)
        with pytest.raises(ValueError, match="requires data.clinical"):
            model(pyg_batch)

    def test_gnn_with_metadata_unknown_gnn_type(self):
        with pytest.raises(ValueError, match="Unknown GNN type"):
            GNN_WithMetadata(gnn_type='NotAGNN', input_dim=1, hidden_dim=64, meta_dim=3)

    def test_forward_graph_only(self, pyg_batch):
        model = GNN_WithMetadata(gnn_type='GCN', input_dim=1, hidden_dim=64, meta_dim=3, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model.forward_graph_only(pyg_batch)
        assert out.shape == (2,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_gnn_clinical_is_alias(self):
        assert GNN_Clinical is GNN_WithMetadata


# =========================================================================
# Multiclass
# =========================================================================

@requires_torch
class TestMulticlass:
    @pytest.mark.parametrize("model_type", ['GCN', 'GINEConv', 'GAT', 'GraphSAGE', 'MLP'])
    def test_multiclass_output_shape(self, pyg_batch, model_type):
        model = get_model(model_type, input_dim=1, hidden_dim=64, num_layers=2, num_classes=4)
        model.eval()
        with torch.no_grad():
            out = model(pyg_batch)
        assert out.shape == (2, 4), f"{model_type}: expected (2,4), got {out.shape}"

    def test_multiclass_cnn(self):
        d = Data(x=torch.randn(10, 1), edge_index=torch.zeros(2, 0, dtype=torch.long))
        batch = Batch.from_data_list([d, d])
        model = CNN_Baseline(input_dim=10, hidden_dim=64, num_classes=4)
        model.eval()
        with torch.no_grad():
            out = model(batch)
        assert out.shape == (2, 4)

    def test_multiclass_metadata_only(self, pyg_batch_with_clinical):
        model = MetadataOnly_Baseline(meta_dim=3, hidden_dim=32, num_classes=4)
        model.eval()
        with torch.no_grad():
            out = model(pyg_batch_with_clinical)
        assert out.shape == (2, 4)


# =========================================================================
# Loss functions — comprehensive
# =========================================================================

@requires_torch
class TestLossFunctions:
    def test_bce_clamped_basic(self):
        loss_fn = get_loss('bce')
        pred = torch.tensor([0.7, 0.3, 0.9])
        target = torch.tensor([1.0, 0.0, 1.0])
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss) and loss.dim() == 0

    def test_bce_clamped_extreme_inputs(self):
        """Inputs of 0.0 and 1.0 must not produce NaN/Inf."""
        loss_fn = get_loss('bce')
        pred = torch.tensor([0.0, 1.0, 0.5])
        target = torch.tensor([0.0, 1.0, 1.0])
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss), f"BCE with extreme inputs = {loss}"

    def test_focal_loss_basic(self):
        loss_fn = get_loss('focal', alpha=0.25, gamma=2.0)
        pred = torch.tensor([0.7, 0.3, 0.9])
        target = torch.tensor([1.0, 0.0, 1.0])
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("reduction", ['mean', 'sum', 'none'])
    def test_focal_loss_reductions(self, reduction):
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction=reduction)
        pred = torch.tensor([0.7, 0.3])
        target = torch.tensor([1.0, 0.0])
        loss = loss_fn(pred, target)
        if reduction == 'none':
            assert loss.shape == (2,)
        else:
            assert loss.dim() == 0

    def test_focal_loss_invalid_reduction(self):
        with pytest.raises(ValueError, match="reduction must be"):
            FocalLoss(reduction='invalid')

    def test_focal_perfect_prediction_low_loss(self):
        loss_fn = FocalLoss(gamma=2.0)
        pred = torch.tensor([0.99, 0.01])
        target = torch.tensor([1.0, 0.0])
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01

    def test_weighted_bce_with_weight(self):
        loss_fn = get_loss('weighted_bce', pos_weight=2.0)
        pred = torch.tensor([0.7, 0.3])
        target = torch.tensor([1.0, 0.0])
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_weighted_bce_without_weight(self):
        loss_fn = get_loss('weighted_bce')
        pred = torch.tensor([0.7, 0.3])
        target = torch.tensor([1.0, 0.0])
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_weighted_bce_compute_pos_weight(self):
        assert WeightedBCELoss.compute_pos_weight([0, 0, 0, 1]) == pytest.approx(3.0)
        assert WeightedBCELoss.compute_pos_weight([0, 1]) == pytest.approx(1.0)

    def test_weighted_bce_degenerate_labels(self):
        assert WeightedBCELoss.compute_pos_weight([1, 1, 1]) == 1.0
        assert WeightedBCELoss.compute_pos_weight([0, 0, 0]) == 1.0

    def test_unknown_loss_raises(self):
        with pytest.raises(ValueError):
            get_loss('nonexistent')

    def test_loss_gradient_flows(self):
        """All losses must allow gradient flow."""
        for loss_name in ['bce', 'focal', 'weighted_bce']:
            loss_fn = get_loss(loss_name, pos_weight=2.0) if loss_name == 'weighted_bce' else get_loss(loss_name)
            pred = torch.tensor([0.6], requires_grad=True)
            target = torch.tensor([1.0])
            loss = loss_fn(pred, target)
            loss.backward()
            assert pred.grad is not None and pred.grad.abs().sum() > 0, \
                f"{loss_name}: no gradient flow"


# =========================================================================
# Factory — comprehensive
# =========================================================================

@requires_torch
class TestGetModel:
    ALL_TYPES = ['GCN', 'GINEConv', 'GAT', 'GraphSAGE', 'EdgeCentricRGCN',
                 'Ensemble', 'MLP', 'CNN', 'GNN_Meta', 'GNN_Clinical', 'MetadataOnly']

    @pytest.mark.parametrize("mtype", ALL_TYPES)
    def test_all_types_instantiate(self, mtype):
        if mtype == 'CNN':
            kwargs = dict(input_dim=100, hidden_dim=64, num_classes=2)
        elif mtype in ('GNN_Meta', 'GNN_Clinical'):
            kwargs = dict(input_dim=1, hidden_dim=64, num_layers=2, num_classes=2, meta_dim=3)
        elif mtype == 'MetadataOnly':
            kwargs = dict(meta_dim=3, hidden_dim=64, num_classes=2)
        else:
            kwargs = dict(input_dim=1, hidden_dim=64, num_layers=2, num_classes=2)
        model = get_model(mtype, **kwargs)
        assert model is not None

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model('NotAModel')

    def test_gat_hidden_dim_not_divisible_by_heads(self):
        with pytest.raises(ValueError, match="divisible by heads"):
            get_model('GAT', input_dim=1, hidden_dim=65, num_layers=2, num_classes=2, heads=4)

    def test_ensemble_hidden_dim_not_divisible_by_4(self):
        with pytest.raises(ValueError, match="divisible by 4"):
            get_model('Ensemble', input_dim=1, hidden_dim=65, num_layers=2, num_classes=2)


# =========================================================================
# EdgeCentricRGCN specifics
# =========================================================================

@requires_torch
class TestEdgeCentricRGCN:
    def test_forward(self, pyg_batch):
        model = EdgeCentricRGCN(input_dim=1, hidden_dim=64, num_classes=2)
        model.eval()
        with torch.no_grad():
            out = model(pyg_batch)
        assert out.shape == (2,)
        assert (out >= 0).all() and (out <= 1).all()

    def test_num_layers_warning(self):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            EdgeCentricRGCN(input_dim=1, hidden_dim=64, num_layers=5, num_classes=2)
            assert any("num_layers=5 ignored" in str(warning.message) for warning in w)

    def test_embedding_shape(self, pyg_batch):
        model = EdgeCentricRGCN(input_dim=1, hidden_dim=64, num_classes=2)
        model.eval()
        with torch.no_grad():
            emb = model.forward_embedding(pyg_batch)
        assert emb.shape == (2, 64)


# =========================================================================
# SklearnModelWrapper
# =========================================================================

@requires_torch
class TestSklearnModelWrapper:
    def test_extract_features_shape(self, pyg_batch):
        features = SklearnModelWrapper.extract_features(pyg_batch)
        assert features.shape == (2, 5)
        assert features.dtype == np.float32

    def test_extract_features_finite(self, pyg_batch):
        features = SklearnModelWrapper.extract_features(pyg_batch)
        assert np.all(np.isfinite(features))

    def test_fit_and_predict_rf(self, pyg_batch):
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        wrapper = SklearnModelWrapper(rf, num_classes=2)

        from torch_geometric.loader import DataLoader
        x = torch.randn(10, 1)
        edge_index = torch.tensor(
            [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9,0]], dtype=torch.long
        )
        edge_attr = torch.rand(10, 1)
        data_list = [
            Data(x=x + i * 0.1, edge_index=edge_index, edge_attr=edge_attr,
                 y=torch.tensor(float(i % 2)))
            for i in range(20)
        ]
        loader = DataLoader(data_list, batch_size=10)
        wrapper.fit_from_loader(loader)

        batch = next(iter(loader))
        out = wrapper(batch)
        assert out.shape[0] == batch.num_graphs
        assert (out >= 0).all() and (out <= 1).all()

    def test_forward_embedding(self, pyg_batch):
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=2, random_state=42)
        wrapper = SklearnModelWrapper(rf, num_classes=2)
        emb = wrapper.forward_embedding(pyg_batch)
        assert emb.shape == (2, 5)


# =========================================================================
# NaN/Inf input detection
# =========================================================================

@requires_torch
class TestNaNInputHandling:
    """Models should produce finite output or raise when fed NaN inputs."""

    @pytest.mark.parametrize("model_type", ["GCN", "GINEConv", "MLP"])
    def test_nan_node_features_detected(self, model_type):
        """Training with NaN features should produce NaN loss (caught by cross_validation)."""
        model = get_model(model_type, input_dim=1, hidden_dim=16, num_layers=2, num_classes=2)
        x = torch.tensor([[float('nan')]] * 10)
        edge_index = torch.tensor(
            [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9,0]], dtype=torch.long
        )
        edge_attr = torch.ones(10, 1)
        batch = torch.zeros(10, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        out = model(data)
        assert torch.isnan(out).any() or not torch.isfinite(out).all(), (
            "Model should propagate NaN from input — silent NaN absorption is dangerous"
        )

    def test_loss_on_nan_predictions(self):
        """Loss functions should not silently produce finite loss from NaN predictions."""
        criterion = get_loss("bce")
        preds = torch.tensor([float('nan'), 0.5])
        targets = torch.tensor([1.0, 0.0])
        try:
            loss = criterion(preds, targets)
            assert not torch.isfinite(loss), "Loss should be non-finite with NaN predictions"
        except RuntimeError:
            pass  # PyTorch raised — also acceptable

    def test_focal_loss_on_nan(self):
        """FocalLoss should not silently produce finite loss from NaN predictions."""
        criterion = get_loss("focal")
        preds = torch.tensor([float('nan'), 0.5])
        targets = torch.tensor([1.0, 0.0])
        try:
            loss = criterion(preds, targets)
            assert not torch.isfinite(loss), "FocalLoss should be non-finite with NaN predictions"
        except RuntimeError:
            pass  # PyTorch raised — also acceptable
