"""Tests for src/config_validation.py — strict config validation."""

import pytest
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config_validation import (
    ConfigError,
    require,
    validate_notebook_config,
    load_and_validate_config,
)


@pytest.fixture
def full_config():
    """Load the actual config.yaml to use as a valid baseline."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class TestRequire:
    def test_simple_key(self):
        assert require({'a': 1}, 'a') == 1

    def test_nested_key(self):
        assert require({'a': {'b': {'c': 42}}}, 'a', 'b', 'c') == 42

    def test_missing_top_level(self):
        with pytest.raises(ConfigError, match="Missing required config key: 'x'"):
            require({'a': 1}, 'x')

    def test_missing_nested(self):
        with pytest.raises(ConfigError, match="Missing required config key: 'a.b.z'"):
            require({'a': {'b': {'c': 1}}}, 'a', 'b', 'z')

    def test_intermediate_not_dict(self):
        with pytest.raises(ConfigError, match="Expected dict"):
            require({'a': 42}, 'a', 'b')

    def test_returns_dict(self):
        result = require({'a': {'b': 1}}, 'a')
        assert result == {'b': 1}

    def test_returns_none_value(self):
        assert require({'a': None}, 'a') is None

    def test_returns_false_value(self):
        assert require({'a': False}, 'a') is False

    def test_returns_zero_value(self):
        assert require({'a': 0}, 'a') == 0


class TestValidateNotebookConfig:
    def test_none_config_raises(self):
        with pytest.raises(ConfigError, match="Config is None"):
            validate_notebook_config(None, 'NB01')

    def test_non_dict_raises(self):
        with pytest.raises(ConfigError, match="Config must be dict"):
            validate_notebook_config("not a dict", 'NB01')

    def test_unknown_notebook_raises(self):
        with pytest.raises(ConfigError, match="Unknown notebook"):
            validate_notebook_config({}, 'NB99')

    def test_empty_config_raises(self):
        with pytest.raises(ConfigError, match="Missing.*required key"):
            validate_notebook_config({}, 'NB01')

    def test_nb01_valid(self, full_config):
        validate_notebook_config(full_config, 'NB01')

    def test_nb02_valid(self, full_config):
        validate_notebook_config(full_config, 'NB02')

    def test_nb03_valid(self, full_config):
        validate_notebook_config(full_config, 'NB03')

    def test_nb04_valid(self, full_config):
        validate_notebook_config(full_config, 'NB04')


class TestMissingKeyDetection:
    """Simulate typos and missing keys that would silently fall back before."""

    def test_typo_in_model_training(self, full_config):
        cfg = dict(full_config)
        cfg['model_trainig'] = cfg.pop('model_training')
        with pytest.raises(ConfigError, match="model_training"):
            validate_notebook_config(cfg, 'NB03')

    def test_missing_hidden_dim(self, full_config):
        import copy
        cfg = copy.deepcopy(full_config)
        del cfg['model_training']['architecture']['hidden_dim']
        with pytest.raises(ConfigError, match="hidden_dim"):
            validate_notebook_config(cfg, 'NB03')

    def test_missing_disease(self, full_config):
        import copy
        cfg = copy.deepcopy(full_config)
        del cfg['data_extraction']['disease_criteria']['disease']
        with pytest.raises(ConfigError, match="disease"):
            validate_notebook_config(cfg, 'NB01')

    def test_missing_graph_type(self, full_config):
        import copy
        cfg = copy.deepcopy(full_config)
        del cfg['graph_construction']['graph_type']
        with pytest.raises(ConfigError, match="graph_type"):
            validate_notebook_config(cfg, 'NB02')

    def test_missing_knn_k(self, full_config):
        import copy
        cfg = copy.deepcopy(full_config)
        del cfg['graph_construction']['knn']['k']
        with pytest.raises(ConfigError, match="k"):
            validate_notebook_config(cfg, 'NB02')

    def test_missing_loss_type(self, full_config):
        """NB03 needs model_training.training.loss.type — not validated at schema level
        but require() would catch it."""
        import copy
        cfg = copy.deepcopy(full_config)
        del cfg['model_training']['training']['loss']['type']
        loss_cfg = require(cfg, 'model_training', 'training', 'loss')
        with pytest.raises(KeyError):
            _ = loss_cfg['type']

    def test_missing_num_classes(self, full_config):
        import copy
        cfg = copy.deepcopy(full_config)
        del cfg['model_training']['architecture']['num_classes']
        with pytest.raises(ConfigError, match="num_classes"):
            validate_notebook_config(cfg, 'NB03')

    def test_missing_edge_construction(self, full_config):
        import copy
        cfg = copy.deepcopy(full_config)
        del cfg['graph_construction']['edge_construction']
        with pytest.raises(ConfigError, match="edge_construction"):
            validate_notebook_config(cfg, 'NB02')

    def test_missing_random_seed(self, full_config):
        import copy
        cfg = copy.deepcopy(full_config)
        del cfg['model_training']['training']['random_seed']
        with pytest.raises(ConfigError, match="random_seed"):
            validate_notebook_config(cfg, 'NB03')

    def test_reports_all_missing_keys(self, full_config):
        import copy
        cfg = copy.deepcopy(full_config)
        del cfg['model_training']['architecture']['hidden_dim']
        del cfg['model_training']['architecture']['num_layers']
        del cfg['model_training']['architecture']['num_classes']
        with pytest.raises(ConfigError, match="Missing 3 required key"):
            validate_notebook_config(cfg, 'NB03')


class TestLoadAndValidateConfig:
    def test_missing_config_raises(self, tmp_path, monkeypatch):
        monkeypatch.delenv('EXPERIMENT_CONFIG_PATH', raising=False)
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError, match="No configuration file found"):
            load_and_validate_config('NB01')

    def test_loads_from_env(self, tmp_path, monkeypatch, full_config):
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(full_config, f)
        monkeypatch.setenv('EXPERIMENT_CONFIG_PATH', str(config_file))
        result = load_and_validate_config('NB01')
        assert result['data_extraction']['disease_criteria']['disease'] == 'IBD'

    def test_loads_from_local_config(self, tmp_path, monkeypatch, full_config):
        monkeypatch.delenv('EXPERIMENT_CONFIG_PATH', raising=False)
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(full_config, f)
        result = load_and_validate_config('NB03')
        assert result['model_training']['architecture']['model_type'] == 'GINEConv'

    def test_empty_yaml_raises(self, tmp_path, monkeypatch):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        monkeypatch.setenv('EXPERIMENT_CONFIG_PATH', str(config_file))
        with pytest.raises(ConfigError, match="empty"):
            load_and_validate_config('NB01')
