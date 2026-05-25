"""Tests for src/cross_validation.py — dataset creation, training, k-fold, FDR."""

import pytest
import numpy as np
from tests.conftest import requires_torch, HAS_TORCH

if HAS_TORCH:
    import torch
    from torch_geometric.data import Data, Batch, Dataset
    from src.cross_validation import (
        create_dataset_from_graphs,
        train_epoch,
        evaluate,
        run_kfold_experiment,
        save_cv_results,
        compare_experiments_with_fdr,
    )
    from src.models import get_model, get_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_toy_graphs(n=40, n_nodes=10, seed=42):
    """Create toy NetworkX-like graphs as PyG Data objects with labels."""
    rng = np.random.default_rng(seed)
    graphs = []
    labels = []
    for i in range(n):
        x = torch.randn(n_nodes, 1)
        edge_index = torch.tensor(
            [[j, (j + 1) % n_nodes] for j in range(n_nodes)]
            + [[(j + 1) % n_nodes, j] for j in range(n_nodes)],
            dtype=torch.long,
        ).t()
        edge_attr = torch.ones(edge_index.shape[1], 1)
        label = int(i < n // 2)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(float(label)))
        graphs.append(data)
        labels.append(label)
    return graphs, np.array(labels)


if HAS_TORCH:
    class ToyDataset(Dataset):
        """Minimal dataset that wraps pre-built Data objects."""

        def __init__(self, data_list, labels):
            super().__init__()
            self._data_list = []
            for d, lab in zip(data_list, labels):
                d_copy = d.clone()
                d_copy.y = torch.tensor(float(lab))
                self._data_list.append(d_copy)

        def len(self):
            return len(self._data_list)

        def get(self, idx):
            return self._data_list[idx]


# ---------------------------------------------------------------------------
# Tests: create_dataset_from_graphs
# ---------------------------------------------------------------------------

@requires_torch
class TestCreateDataset:
    def test_basic(self):
        graphs, labels = _make_toy_graphs(10)
        ds = create_dataset_from_graphs(graphs, labels, ToyDataset)
        assert len(ds) == 10

    def test_labels_match(self):
        graphs, labels = _make_toy_graphs(10)
        ds = create_dataset_from_graphs(graphs, labels, ToyDataset)
        for i in range(len(ds)):
            assert ds[i].y.item() == float(labels[i])


# ---------------------------------------------------------------------------
# Tests: train_epoch / evaluate
# ---------------------------------------------------------------------------

@requires_torch
class TestTrainEpoch:
    def test_returns_loss_and_accuracy(self):
        from torch_geometric.loader import DataLoader

        graphs, labels = _make_toy_graphs(20)
        ds = ToyDataset(graphs, labels)
        loader = DataLoader(ds, batch_size=10, shuffle=False)

        model = get_model("GCN", input_dim=1, hidden_dim=16, num_layers=2, num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = get_loss("bce")

        loss, acc = train_epoch(model, loader, optimizer, criterion, "cpu")
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert np.isfinite(loss)
        assert 0.0 <= acc <= 1.0


@requires_torch
class TestEvaluate:
    def test_returns_expected_tuple(self):
        from torch_geometric.loader import DataLoader

        graphs, labels = _make_toy_graphs(20)
        ds = ToyDataset(graphs, labels)
        loader = DataLoader(ds, batch_size=10, shuffle=False)

        model = get_model("GCN", input_dim=1, hidden_dim=16, num_layers=2, num_classes=2)
        criterion = get_loss("bce")

        result = evaluate(model, loader, criterion, "cpu")
        assert len(result) == 6
        val_loss, val_acc, preds, labs, probs, balanced_acc = result
        assert np.isfinite(val_loss)
        assert 0.0 <= val_acc <= 1.0
        assert len(preds) == 20
        assert len(labs) == 20
        assert 0.0 <= balanced_acc <= 1.0


# ---------------------------------------------------------------------------
# Tests: run_kfold_experiment input validation
# ---------------------------------------------------------------------------

@requires_torch
class TestRunKfoldValidation:
    def test_empty_graphs_raises(self):
        with pytest.raises(ValueError, match="empty"):
            run_kfold_experiment(
                graphs=[], labels=np.array([]),
                model_class=None, model_params={},
                training_params={}, dataset_class=ToyDataset,
            )

    def test_mismatched_lengths_raises(self):
        graphs, labels = _make_toy_graphs(20)
        with pytest.raises(ValueError, match="same length"):
            run_kfold_experiment(
                graphs=graphs, labels=labels[:10],
                model_class=None, model_params={},
                training_params={}, dataset_class=ToyDataset,
            )

    def test_single_class_raises(self):
        graphs, _ = _make_toy_graphs(20)
        labels = np.zeros(20)
        with pytest.raises(ValueError, match="at least 2 classes"):
            run_kfold_experiment(
                graphs=graphs, labels=labels,
                model_class=None, model_params={},
                training_params={}, dataset_class=ToyDataset,
            )

    def test_nfolds_too_small_raises(self):
        graphs, labels = _make_toy_graphs(20)
        with pytest.raises(ValueError, match="n_folds must be >= 2"):
            run_kfold_experiment(
                graphs=graphs, labels=labels,
                model_class=None, model_params={},
                training_params={}, n_folds=1,
                dataset_class=ToyDataset,
            )

    def test_nfolds_exceeds_samples_raises(self):
        graphs, labels = _make_toy_graphs(4)
        with pytest.raises(ValueError, match="cannot exceed"):
            run_kfold_experiment(
                graphs=graphs, labels=labels,
                model_class=None, model_params={},
                training_params={}, n_folds=10,
                dataset_class=ToyDataset,
            )

    def test_no_dataset_class_raises(self):
        graphs, labels = _make_toy_graphs(20)
        with pytest.raises(ValueError, match="dataset_class"):
            run_kfold_experiment(
                graphs=graphs, labels=labels,
                model_class=None, model_params={},
                training_params={},
            )


# ---------------------------------------------------------------------------
# Tests: compare_experiments_with_fdr
# ---------------------------------------------------------------------------

class TestCompareExperiments:
    def test_basic_comparison(self):
        results = {
            'baseline': {'accuracies': [0.6, 0.62, 0.58, 0.61, 0.59]},
            'model_a': {'accuracies': [0.7, 0.72, 0.68, 0.71, 0.69]},
        }
        try:
            out = compare_experiments_with_fdr(results)
        except ImportError:
            pytest.skip("statsmodels not installed")

        assert 'comparisons' in out
        assert len(out['comparisons']) == 1
        assert out['comparisons'][0] == 'model_a'
        assert len(out['raw_p_values']) == 1

    def test_missing_baseline_returns_error(self):
        results = {'model_a': {'accuracies': [0.7]}}
        out = compare_experiments_with_fdr(results)
        assert 'error' in out

    def test_mismatched_lengths_skips(self):
        results = {
            'baseline': {'accuracies': [0.6, 0.62, 0.58]},
            'model_a': {'accuracies': [0.7, 0.72]},
        }
        try:
            out = compare_experiments_with_fdr(results)
        except ImportError:
            pytest.skip("statsmodels not installed")
        assert 'error' in out or len(out.get('comparisons', [])) == 0


# ---------------------------------------------------------------------------
# Tests: save_cv_results
# ---------------------------------------------------------------------------

@requires_torch
class TestSaveCvResults:
    def test_saves_json(self, tmp_path):
        import json

        results = {
            'fold_results': [
                {
                    'fold_idx': 0,
                    'mean_accuracy': np.float64(0.85),
                    'runs': [{'model_state': {'key': torch.tensor([1.0])}}],
                }
            ],
            'aggregated': {'mean_accuracy': 0.85},
        }
        out_path = tmp_path / "cv_results.json"
        save_cv_results(results, out_path)

        assert out_path.exists()
        with open(out_path) as f:
            data = json.load(f)
        assert data['aggregated']['mean_accuracy'] == 0.85
        assert 'model_state' not in data['fold_results'][0]['runs'][0]


# ---------------------------------------------------------------------------
# Leakage / negative-control regression tests
# ---------------------------------------------------------------------------

@requires_torch
class TestNoDataLeakage:
    """If labels carry no signal, held-out test accuracy must be ~chance.
    A high score here would mean train/test information is leaking (the silent
    failure mode that has burned this project before)."""

    def test_random_labels_give_chance_accuracy(self):
        from src.models import GNN_GCN
        graphs, _ = _make_toy_graphs(n=60, n_nodes=10, seed=1)
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, size=len(graphs))
        labels[0], labels[1] = 0, 1  # guarantee both classes
        results = run_kfold_experiment(
            graphs, labels,
            model_class=GNN_GCN,
            model_params={'input_dim': 1, 'hidden_dim': 8, 'num_layers': 2,
                          'dropout': 0.0, 'pooling': 'mean', 'num_classes': 2},
            training_params={'batch_size': 8, 'num_epochs': 10, 'learning_rate': 0.01,
                             'weight_decay': 0.0, 'early_stopping_patience': 5,
                             'loss': {'type': 'bce'}},
            dataset_class=ToyDataset,
            n_folds=3, num_runs_per_fold=1, random_seed=0, verbose=False,
        )
        mean_acc = results['aggregated']['mean_accuracy']
        # Leakage would drive this toward ~1.0; random labels must stay near chance.
        assert mean_acc < 0.80, (
            f"Held-out test accuracy {mean_acc:.3f} on RANDOM labels is implausibly "
            "high — indicates train/test leakage."
        )

    def test_separable_signal_is_learnable(self):
        """Sanity counterpart: a planted, separable signal should beat chance,
        confirming the pipeline can actually learn (so the leakage test above is
        meaningful rather than trivially passing because nothing trains)."""
        import torch
        from torch_geometric.data import Data
        from src.models import GNN_GCN
        rng = np.random.default_rng(7)
        graphs, labels = [], []
        for i in range(60):
            label = int(i < 30)
            # class signal in node features: shift the mean by class
            x = torch.randn(10, 1) + (2.0 if label else -2.0)
            ei = torch.tensor([[j, (j + 1) % 10] for j in range(10)]
                              + [[(j + 1) % 10, j] for j in range(10)], dtype=torch.long).t()
            ea = torch.ones(ei.shape[1], 1)
            graphs.append(Data(x=x, edge_index=ei, edge_attr=ea, y=torch.tensor(float(label))))
            labels.append(label)
        results = run_kfold_experiment(
            graphs, np.array(labels),
            model_class=GNN_GCN,
            model_params={'input_dim': 1, 'hidden_dim': 8, 'num_layers': 2,
                          'dropout': 0.0, 'pooling': 'mean', 'num_classes': 2},
            training_params={'batch_size': 8, 'num_epochs': 30, 'learning_rate': 0.01,
                             'weight_decay': 0.0, 'early_stopping_patience': 10,
                             'loss': {'type': 'bce'}},
            dataset_class=ToyDataset,
            n_folds=3, num_runs_per_fold=1, random_seed=0, verbose=False,
        )
        assert results['aggregated']['mean_accuracy'] > 0.75, (
            "Pipeline failed to learn a clearly separable signal — training is broken."
        )
