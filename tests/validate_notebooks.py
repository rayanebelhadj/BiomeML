#!/usr/bin/env python3
"""
Static validation of notebook configuration and setup.

Checks performed (no execution, pure AST/string analysis):
  1. Config structure — required keys present in config.yaml / experiments.yaml
  2. Model type coverage — every model_type in experiments.yaml has a handler
  3. Graph type coverage — every graph_type in experiments.yaml has a builder
  4. Weight strategy coverage — every weight_transform is a known strategy
  5. Import consistency — notebooks import functions that exist in src/
  6. Notebook cell ordering — critical cells exist and are in the right order
  7. Disease path consistency — paths reference {DISEASE} correctly
  8. Bug-pattern detection — known anti-patterns that caused past bugs

Run as:  python -m pytest tests/validate_notebooks.py -v
    or:  python tests/validate_notebooks.py
"""

import json
import re
import os
import sys
import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _load_notebook(path) -> dict:
    with open(path) as f:
        return json.load(f)


def _notebook_source(nb: dict) -> str:
    """Concatenate all code cell sources into a single string."""
    parts = []
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            parts.append(''.join(cell.get('source', [])))
    return '\n'.join(parts)


def _notebook_cell_sources(nb: dict):
    """Return list of (idx, source_str) for code cells."""
    out = []
    for i, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            out.append((i, ''.join(cell.get('source', []))))
    return out


# ---------------------------------------------------------------------------
# 1. config.yaml structure
# ---------------------------------------------------------------------------

class TestConfigYaml:
    @pytest.fixture(autouse=True)
    def load(self):
        self.path = ROOT / 'config.yaml'
        if not self.path.exists():
            pytest.skip("config.yaml not found")
        self.cfg = _load_yaml(self.path)

    def test_has_graph_construction(self):
        assert 'graph_construction' in self.cfg

    def test_has_model_training(self):
        assert 'model_training' in self.cfg

    def test_has_data_extraction(self):
        assert 'data_extraction' in self.cfg

    def test_knn_params_present(self):
        knn = self.cfg.get('graph_construction', {}).get('knn', {})
        assert 'k' in knn, "graph_construction.knn.k missing"
        assert 'symmetric' in knn, "graph_construction.knn.symmetric missing"

    def test_knn_k_positive(self):
        k = self.cfg.get('graph_construction', {}).get('knn', {}).get('k')
        if k is not None:
            assert k >= 1, f"knn.k must be >= 1, got {k}"

    def test_training_params_present(self):
        mt = self.cfg.get('model_training', {})
        assert 'training' in mt or 'architecture' in mt

    def test_loss_type_valid(self):
        loss = (self.cfg.get('model_training', {})
                .get('training', {})
                .get('loss', {}))
        if isinstance(loss, dict) and 'type' in loss:
            valid = {'bce', 'focal', 'weighted_bce', 'cross_entropy'}
            assert loss['type'] in valid, f"Unknown loss type: {loss['type']}"

    def test_quality_min_nodes_positive(self):
        q = self.cfg.get('graph_construction', {}).get('quality', {})
        if 'min_nodes' in q:
            assert q['min_nodes'] > 0
        if 'min_edges' in q:
            assert q['min_edges'] > 0

    def test_learning_rate_range(self):
        lr = (self.cfg.get('model_training', {})
              .get('training', {})
              .get('learning_rate'))
        if lr is not None:
            assert 0 < lr < 1, f"learning_rate {lr} seems wrong"

    def test_dropout_range(self):
        dr = (self.cfg.get('model_training', {})
              .get('architecture', {})
              .get('dropout'))
        if dr is not None:
            assert 0 <= dr < 1, f"dropout {dr} out of [0, 1)"

    def test_hidden_dim_positive(self):
        hd = (self.cfg.get('model_training', {})
              .get('architecture', {})
              .get('hidden_dim'))
        if hd is not None:
            assert hd > 0 and hd % 2 == 0, f"hidden_dim {hd} must be positive and even"


# ---------------------------------------------------------------------------
# 2. experiments.yaml — comprehensive
# ---------------------------------------------------------------------------

class TestExperimentsYaml:
    @pytest.fixture(autouse=True)
    def load(self):
        self.path = ROOT / 'experiments.yaml'
        if not self.path.exists():
            pytest.skip("experiments.yaml not found")
        self.exps = _load_yaml(self.path)

    def _all_model_types(self):
        types = set()
        for exp in self.exps.values():
            mt = (exp.get('model_training', {})
                  .get('architecture', {})
                  .get('model_type'))
            if mt:
                types.add(mt)
        return types

    def _all_graph_types(self):
        types = set()
        for exp in self.exps.values():
            gt = exp.get('graph_construction', {}).get('graph_type')
            if gt:
                types.add(gt)
        return types

    def _all_weight_transforms(self):
        transforms = set()
        for exp in self.exps.values():
            wt = (exp.get('graph_construction', {})
                  .get('weights', {})
                  .get('weight_transform'))
            if wt:
                transforms.add(wt)
        return transforms

    def test_model_types_known(self):
        known = {
            'GCN', 'GINEConv', 'GAT', 'GraphSAGE', 'EdgeCentricRGCN',
            'Ensemble', 'MLP', 'CNN', 'GNN_Meta', 'GNN_Clinical',
            'MetadataOnly', 'RandomForest', 'XGBoost',
        }
        for mt in self._all_model_types():
            assert mt in known, f"experiments.yaml references unknown model_type: {mt}"

    def test_graph_types_known(self):
        known = {'knn', 'mst', 'threshold', 'tree', 'hierarchical'}
        for gt in self._all_graph_types():
            assert gt in known, f"experiments.yaml references unknown graph_type: {gt}"

    def test_weight_transforms_known(self):
        known = {
            'identity', 'inverse', 'exponential', 'exp', 'binary', 'linear',
            'abundance_product', 'abundance_geometric', 'abundance_log',
            'abundance_min', 'abundance_max',
        }
        for wt in self._all_weight_transforms():
            assert wt in known, f"experiments.yaml references unknown weight_transform: {wt}"

    def test_every_experiment_has_disease(self):
        for name, exp in self.exps.items():
            assert 'disease' in exp, f"Experiment '{name}' missing 'disease' key"

    def test_every_experiment_has_name(self):
        for key, exp in self.exps.items():
            assert 'name' in exp, f"Experiment '{key}' missing 'name' key"

    def test_no_duplicate_names(self):
        names = [exp.get('name') for exp in self.exps.values() if 'name' in exp]
        assert len(names) == len(set(names)), f"Duplicate experiment names: {[n for n in names if names.count(n) > 1]}"

    def test_hidden_dim_divisible_by_4_for_gat_ensemble(self):
        for name, exp in self.exps.items():
            mt = exp.get('model_training', {}).get('architecture', {}).get('model_type')
            hd = exp.get('model_training', {}).get('architecture', {}).get('hidden_dim')
            if mt in ('GAT', 'Ensemble') and hd is not None:
                assert hd % 4 == 0, (
                    f"Experiment '{name}': {mt} requires hidden_dim divisible by 4, got {hd}"
                )

    def test_knn_k_in_range(self):
        for name, exp in self.exps.items():
            k = exp.get('graph_construction', {}).get('knn', {}).get('k')
            if k is not None:
                assert 1 <= k <= 100, f"Experiment '{name}': knn.k={k} out of [1,100]"


# ---------------------------------------------------------------------------
# 3. Notebook 02 — graph construction
# ---------------------------------------------------------------------------

class TestNotebook02:
    @pytest.fixture(autouse=True)
    def load(self):
        self.path = ROOT / 'notebooks' / '02_graph_construction.ipynb'
        if not self.path.exists():
            pytest.skip("02_graph_construction.ipynb not found")
        self.nb = _load_notebook(self.path)
        self.src = _notebook_source(self.nb)

    def test_imports_graph_utils(self):
        assert 'from src.graph_utils import' in self.src or 'src.graph_utils' in self.src

    def test_handles_all_graph_types(self):
        for gt in ['knn', 'mst', 'threshold', 'tree']:
            assert gt in self.src, f"Notebook 02 missing handler for graph_type='{gt}'"

    def test_hierarchical_blocked_or_handled(self):
        assert 'hierarchical' in self.src

    def test_graph_params_defaults(self):
        assert 'GRAPH_PARAMS' in self.src

    def test_graph_type_overridden_from_config(self):
        assert "graph_type" in self.src

    def test_saves_graphs_and_labels(self):
        assert 'nx_graphs_' in self.src
        assert 'labels_' in self.src

    def test_weight_transform_applied(self):
        assert 'weight_transform' in self.src or 'weight_strategy' in self.src

    def test_validates_min_nodes_and_edges(self):
        assert 'min_nodes' in self.src
        assert 'min_edges' in self.src

    def test_removes_isolates_or_zero_abundance_nodes(self):
        assert 'remove_isolates' in self.src or 'isolat' in self.src or 'abundance_dict' in self.src

    def test_checkpoint_support(self):
        assert 'checkpoint' in self.src.lower()


# ---------------------------------------------------------------------------
# 4. Notebook 03 — model training
# ---------------------------------------------------------------------------

class TestNotebook03:
    @pytest.fixture(autouse=True)
    def load(self):
        self.path = ROOT / 'notebooks' / '03_model_training.ipynb'
        if not self.path.exists():
            pytest.skip("03_model_training.ipynb not found")
        self.nb = _load_notebook(self.path)
        self.src = _notebook_source(self.nb)

    def test_imports_models(self):
        assert 'src.models' in self.src or 'from src.models' in self.src

    def test_loads_experiment_config(self):
        assert 'EXPERIMENT_CONFIG' in self.src

    def test_model_type_dispatched(self):
        assert 'model_type' in self.src

    def test_handles_metadata_only(self):
        assert 'MetadataOnly' in self.src

    def test_handles_gnn_meta(self):
        assert 'GNN_Meta' in self.src or 'GNN_WithMetadata' in self.src

    def test_handles_sklearn_models(self):
        assert ('RandomForest' in self.src or 'XGBoost' in self.src
                or 'SklearnModelWrapper' in self.src or 'sklearn' in self.src)

    def test_early_stopping_uses_deep_copy(self):
        assert 'deepcopy' in self.src or 'detach().clone()' in self.src or 'copy.deepcopy' in self.src

    def test_saves_model(self):
        assert 'model.pt' in self.src or 'save' in self.src

    def test_saves_results_json(self):
        assert 'results.json' in self.src or 'json.dump' in self.src

    def test_clinical_dim_passed_to_model(self):
        assert 'clinical_dim' in self.src or 'meta_dim' in self.src

    def test_no_bare_except(self):
        cells = _notebook_cell_sources(self.nb)
        for idx, source in cells:
            lines = source.split('\n')
            for line_no, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped == 'except:':
                    pytest.fail(
                        f"Cell {idx}, line {line_no}: bare `except:` found. "
                        "Use `except Exception:` instead."
                    )

    def test_clinical_unsqueeze_not_flatten(self):
        """The critical bug: clinical_tensor.flatten() → wrong PyG batching.
        Must use unsqueeze(0)."""
        if 'clinical_tensor' in self.src:
            assert 'clinical_tensor.flatten()' not in self.src, \
                "BUG: clinical_tensor.flatten() → PyG concatenates instead of stacking. Use unsqueeze(0)."

    def test_best_model_state_not_shared_reference(self):
        """model.state_dict() returns a reference that Adam can corrupt in-place.
        Must use deepcopy or detach().clone()."""
        if 'best_model_state' in self.src:
            assert ('deepcopy' in self.src or 'detach().clone()' in self.src), \
                "best_model_state must use deepcopy/detach().clone(), not bare state_dict()"

    def test_gradient_clipping_present(self):
        assert 'clip_grad_norm' in self.src or 'clip_grad_value' in self.src or 'max_norm' in self.src

    def test_stratified_split(self):
        assert 'stratif' in self.src.lower() or 'Stratified' in self.src


# ---------------------------------------------------------------------------
# 5. Notebook 01 — data extraction
# ---------------------------------------------------------------------------

class TestNotebook01:
    @pytest.fixture(autouse=True)
    def load(self):
        self.path = ROOT / 'notebooks' / '01_data_extraction.ipynb'
        if not self.path.exists():
            pytest.skip("01_data_extraction.ipynb not found")
        self.nb = _load_notebook(self.path)
        self.src = _notebook_source(self.nb)

    def test_loads_biom(self):
        assert 'biom' in self.src.lower() or 'BIOM' in self.src

    def test_loads_metadata(self):
        assert 'metadata' in self.src.lower()

    def test_case_control_matching(self):
        assert 'match' in self.src.lower() or 'case' in self.src.lower()

    def test_saves_distance_matrices(self):
        assert 'MATRICES_' in self.src or 'pickle' in self.src.lower()

    def test_saves_tsv(self):
        assert '.tsv' in self.src or 'to_csv' in self.src

    def test_no_bare_except(self):
        cells = _notebook_cell_sources(self.nb)
        for idx, source in cells:
            for line_no, line in enumerate(source.split('\n'), 1):
                if line.strip() == 'except:':
                    pytest.fail(f"NB01 Cell {idx}, line {line_no}: bare `except:`")


# ---------------------------------------------------------------------------
# 6. Notebook 04 — model interpretation (if exists)
# ---------------------------------------------------------------------------

class TestNotebook04:
    @pytest.fixture(autouse=True)
    def load(self):
        self.path = ROOT / 'notebooks' / '04_model_interpretation.ipynb'
        if not self.path.exists():
            pytest.skip("04_model_interpretation.ipynb not found")
        self.nb = _load_notebook(self.path)
        self.src = _notebook_source(self.nb)

    def test_imports_interpretation_module(self):
        assert 'model_interpretation' in self.src

    def test_handles_gradient_method(self):
        assert 'gradient' in self.src.lower() or 'integrated_gradient' in self.src

    def test_handles_embedding_visualization(self):
        assert 'tsne' in self.src.lower() or 't-SNE' in self.src or 'umap' in self.src.lower()

    def test_model_classes_dict_complete(self):
        """MODEL_CLASSES dict must include all common model types."""
        if 'MODEL_CLASSES' in self.src:
            for mt in ['GINEConv', 'GCN', 'GAT', 'GraphSAGE', 'MLP']:
                assert f"'{mt}'" in self.src or f'"{mt}"' in self.src, \
                    f"MODEL_CLASSES missing '{mt}'"

    def test_no_bare_except(self):
        cells = _notebook_cell_sources(self.nb)
        for idx, source in cells:
            for line_no, line in enumerate(source.split('\n'), 1):
                if line.strip() == 'except:':
                    pytest.fail(f"NB04 Cell {idx}, line {line_no}: bare `except:`")


# ---------------------------------------------------------------------------
# 7. Anti-pattern detection across all notebooks
# ---------------------------------------------------------------------------

class TestAntiPatterns:
    """Detect known bug patterns that caused issues in the past."""

    @pytest.fixture(autouse=True)
    def load_all(self):
        self.notebooks = {}
        nb_dir = ROOT / 'notebooks'
        if nb_dir.exists():
            for f in sorted(nb_dir.glob('*.ipynb')):
                self.notebooks[f.stem] = _notebook_source(_load_notebook(f))

    def test_no_bare_except_anywhere(self):
        for name, src in self.notebooks.items():
            for i, line in enumerate(src.split('\n'), 1):
                if line.strip() == 'except:':
                    pytest.fail(f"{name} line {i}: bare `except:` — use `except Exception:`")

    def test_no_model_state_dict_without_copy(self):
        """best_model_state = model.state_dict() is a bug (shared reference)."""
        pattern = re.compile(r'best.*state\s*=\s*model\.state_dict\(\)\s*$', re.MULTILINE)
        for name, src in self.notebooks.items():
            if pattern.search(src):
                pytest.fail(
                    f"{name}: `best_model_state = model.state_dict()` found. "
                    "This is a shared reference — use deepcopy or detach().clone()."
                )

    def test_no_flatten_for_clinical(self):
        for name, src in self.notebooks.items():
            if 'clinical_tensor.flatten()' in src:
                pytest.fail(
                    f"{name}: `clinical_tensor.flatten()` found. "
                    "Use `unsqueeze(0)` for correct PyG batching."
                )

    def test_no_silent_graph_type_fallback(self):
        """graph_type must never silently change (e.g. tree→kNN)."""
        for name, src in self.notebooks.items():
            if 'falling back to k-NN' in src or "fallback to knn" in src.lower():
                pytest.fail(f"{name}: silent graph_type fallback detected (tree→kNN)")

    def test_no_silent_weight_transform_fallback(self):
        for name, src in self.notebooks.items():
            if 'falling back to identity' in src.lower():
                pytest.fail(f"{name}: silent weight_transform fallback to identity")

    def test_no_silent_model_type_fallback(self):
        for name, src in self.notebooks.items():
            if 'falling back to GINEConv' in src or 'falling back to legacy' in src:
                pytest.fail(f"{name}: silent model_type fallback detected")

    def test_no_np_random_seed_in_training_loop(self):
        """np.random.seed inside a training loop resets randomness each iteration."""
        for name, src in self.notebooks.items():
            if 'for epoch' in src and 'np.random.seed' in src:
                lines = src.split('\n')
                in_loop = False
                for line in lines:
                    if 'for epoch' in line:
                        in_loop = True
                    if in_loop and 'np.random.seed' in line:
                        pytest.fail(f"{name}: np.random.seed inside training loop")
                    if in_loop and (line.strip() and not line.startswith(' ') and 'for' not in line):
                        in_loop = False

    def test_no_except_without_raise(self):
        """Every except block should re-raise or the error is silently swallowed."""
        import re
        allowlist = {
            'numpy._core',  # numpy compat shim
            'checkpoint_data.pkl',  # checkpoint loading on first run
        }
        for name, src in self.notebooks.items():
            for m in re.finditer(
                r'except\s+(?:Exception|RuntimeError|ValueError|KeyError|TypeError)'
                r'(?:\s+as\s+\w+)?:\s*\n((?:[ \t]+[^\n]*\n)*)',
                src,
            ):
                block = m.group(1)
                if 'raise' in block:
                    continue
                if any(kw in block for kw in allowlist):
                    continue
                if 'pass' in block.split('\n')[0].strip():
                    if any(kw in src[max(0, m.start()-200):m.start()] for kw in allowlist):
                        continue
                snippet = block.strip().split('\n')[0][:80]
                pytest.fail(
                    f"{name}: except block without raise detected – "
                    f"first line: `{snippet}`"
                )

    def test_no_variable_set_to_none_on_failure(self):
        """Variables should not be set to None in except blocks (masks errors)."""
        import re
        for name, src in self.notebooks.items():
            for m in re.finditer(
                r'except\s+\w+.*?:\s*\n((?:[ \t]+[^\n]*\n)*)',
                src,
            ):
                block = m.group(1)
                if 'raise' in block:
                    continue
                if re.search(r'=\s*None', block):
                    snippet = block.strip().split('\n')[0][:80]
                    pytest.fail(
                        f"{name}: variable set to None in except block – "
                        f"first line: `{snippet}`"
                    )

    def test_no_experiment_config_none_fallback(self):
        """EXPERIMENT_CONFIG should never be set to None or {} as a fallback."""
        for name, src in self.notebooks.items():
            for i, line in enumerate(src.split('\n'), 1):
                stripped = line.strip()
                if stripped in ('EXPERIMENT_CONFIG = None', 'EXPERIMENT_CONFIG = {}'):
                    pytest.fail(
                        f"{name} line {i}: `{stripped}` — config must always be loaded, "
                        f"never set to None/empty as fallback"
                    )

    def test_no_using_default_parameters_message(self):
        """Notebooks must not print 'using default parameters' — all params come from config."""
        for name, src in self.notebooks.items():
            lower_src = src.lower()
            if 'using default parameters' in lower_src or 'using default settings' in lower_src:
                pytest.fail(
                    f"{name}: found 'using default parameters/settings' message — "
                    f"all parameters must come from config, no fallback defaults"
                )

    def test_config_loaded_via_validation_module(self):
        """Each notebook should use load_and_validate_config or validate_notebook_config."""
        for name, src in self.notebooks.items():
            if 'load_and_validate_config' in src or 'validate_notebook_config' in src:
                continue
            if 'EXPERIMENT_CONFIG' in src:
                pytest.fail(
                    f"{name}: uses EXPERIMENT_CONFIG but does not call "
                    f"load_and_validate_config or validate_notebook_config"
                )


# ---------------------------------------------------------------------------
# 8. Cross-module consistency
# ---------------------------------------------------------------------------

class TestCrossModuleConsistency:
    def test_get_model_covers_experiments_yaml(self):
        exp_path = ROOT / 'experiments.yaml'
        if not exp_path.exists():
            pytest.skip("experiments.yaml not found")

        exps = _load_yaml(exp_path)
        skip = {'RandomForest', 'XGBoost'}

        try:
            from src.models import get_model
        except ImportError:
            pytest.skip("Cannot import src.models (torch not available)")

        model_types = set()
        for exp in exps.values():
            mt = exp.get('model_training', {}).get('architecture', {}).get('model_type')
            if mt and mt not in skip:
                model_types.add(mt)

        for mt in model_types:
            try:
                get_model(mt, input_dim=1, hidden_dim=64, num_layers=2, num_classes=2)
            except Exception as e:
                pytest.fail(f"get_model('{mt}') failed: {e}")

    def test_edge_weight_strategies_cover_experiments(self):
        exp_path = ROOT / 'experiments.yaml'
        if not exp_path.exists():
            pytest.skip("experiments.yaml not found")

        exps = _load_yaml(exp_path)

        try:
            from src.edge_weights import EDGE_WEIGHT_STRATEGIES
        except ImportError:
            pytest.skip("Cannot import src.edge_weights")

        simple_transforms = {'identity', 'inverse', 'exponential', 'exp', 'binary', 'linear'}
        known = set(EDGE_WEIGHT_STRATEGIES.keys()) | simple_transforms

        for name, exp in exps.items():
            wt = exp.get('graph_construction', {}).get('weights', {}).get('weight_transform')
            if wt:
                assert wt in known, f"Experiment '{name}' uses weight_transform '{wt}' not in known strategies"

    def test_graph_builders_cover_experiments(self):
        exp_path = ROOT / 'experiments.yaml'
        if not exp_path.exists():
            pytest.skip("experiments.yaml not found")

        exps = _load_yaml(exp_path)

        from src.graph_utils import get_graph_builder
        for name, exp in exps.items():
            gt = exp.get('graph_construction', {}).get('graph_type')
            if gt:
                try:
                    get_graph_builder(gt)
                except ValueError:
                    pytest.fail(f"Experiment '{name}' uses graph_type '{gt}' not supported by get_graph_builder")

    def test_config_yaml_and_experiments_yaml_compatible(self):
        """All experiments should be runnable against the base config structure."""
        cfg_path = ROOT / 'config.yaml'
        exp_path = ROOT / 'experiments.yaml'
        if not cfg_path.exists() or not exp_path.exists():
            pytest.skip("config files not found")

        cfg = _load_yaml(cfg_path)
        exps = _load_yaml(exp_path)

        base_sections = set(cfg.keys())
        for name, exp in exps.items():
            for key in exp:
                if key not in ('name', 'disease', 'description', 'n_runs'):
                    assert key in base_sections, \
                        f"Experiment '{name}' has section '{key}' not in base config.yaml"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
