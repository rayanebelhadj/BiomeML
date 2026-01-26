"""Strict configuration validation for the BiomeML pipeline.

Every notebook MUST load and validate the experiment config before using it.
When EXPERIMENT_CONFIG_PATH is set (automated run), the merged config is
guaranteed to contain every key from config.yaml plus experiment overrides.
This module validates that assumption and crashes immediately if any required
key is missing.

Usage in notebooks:
    from config_validation import validate_notebook_config, require

    validate_notebook_config(EXPERIMENT_CONFIG, 'NB03')
    model_type = require(EXPERIMENT_CONFIG, 'model_training', 'architecture', 'model_type')
"""

from pathlib import Path
from typing import Any, Dict, List


class ConfigError(Exception):
    """Raised when a required config key is missing or has wrong type."""
    pass


def require(config: dict, *keys: str) -> Any:
    """Navigate a nested config dict and return the value, raising on any missing key.

    >>> require({'a': {'b': 1}}, 'a', 'b')
    1
    >>> require({'a': {}}, 'a', 'b')  # raises ConfigError
    """
    current = config
    path_so_far: List[str] = []
    for key in keys:
        path_so_far.append(str(key))
        if not isinstance(current, dict):
            raise ConfigError(
                f"Expected dict at '{'.'.join(path_so_far[:-1])}', "
                f"got {type(current).__name__}"
            )
        if key not in current:
            raise ConfigError(
                f"Missing required config key: '{'.'.join(path_so_far)}'"
            )
        current = current[key]
    return current


def load_and_validate_config(notebook: str) -> dict:
    """Load the experiment config from EXPERIMENT_CONFIG_PATH or config.yaml,
    validate it for the given notebook, and return it.

    Raises FileNotFoundError if no config is found.
    Raises ConfigError if required keys are missing.
    """
    import os
    import yaml

    config_path = None

    if 'EXPERIMENT_CONFIG_PATH' in os.environ:
        config_path = Path(os.environ['EXPERIMENT_CONFIG_PATH'])
    else:
        candidates = [
            Path("config.yaml"),
            Path("../config.yaml"),
            Path("../../config.yaml"),
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if config_path is None or not config_path.exists():
        raise FileNotFoundError(
            "No configuration file found. "
            "Set EXPERIMENT_CONFIG_PATH or place config.yaml in the project root."
        )

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config is None or not isinstance(config, dict):
        raise ConfigError(f"Config file {config_path} is empty or not a YAML dict")

    print(f"Loaded config from: {config_path}")
    validate_notebook_config(config, notebook)
    return config


_NB01_REQUIRED = {
    'data_extraction': {
        'disease_criteria': ['disease'],
        'feature_filtering': ['enable', 'min_prevalence', 'min_total_abundance', 'max_features'],
        'distance_matrices': ['primary_matrix', 'use_tree_distance', 'use_sequence_distance', 'use_graph_distance'],
        'matching': ['random_seed'],
        'output': ['base_dir'],
    },
}

_NB02_REQUIRED = {
    'data_extraction': {
        'disease_criteria': ['disease'],
    },
    'graph_construction': {
        'graph_type': [],
        'knn': ['k', 'symmetric', 'max_distance_factor'],
        'quality': ['min_nodes', 'min_edges', 'max_nodes_per_graph', 'remove_isolates', 'connectivity_check'],
        'weights': ['normalize', 'weight_transform'],
        'processing': ['batch_size', 'save_intermediate'],
        'edge_construction': ['randomize_edges', 'preserve_degree'],
    },
}

_NB03_REQUIRED = {
    'data_extraction': {
        'disease_criteria': ['disease'],
    },
    'model_training': {
        'architecture': ['model_type', 'hidden_dim', 'num_layers', 'dropout', 'pooling', 'num_classes'],
        'training': ['batch_size', 'num_epochs', 'learning_rate', 'weight_decay', 'patience', 'random_seed', 'loss'],
        'dataset': ['test_size', 'val_size', 'random_seed'],
        'evaluation': ['enable_cross_validation', 'n_folds', 'num_runs_per_experiment'],
    },
}

_NB04_REQUIRED = {
    'data_extraction': {
        'disease_criteria': ['disease'],
    },
    'model_training': {
        'architecture': ['model_type', 'hidden_dim', 'num_layers', 'dropout', 'pooling', 'num_classes'],
    },
}

_REQUIRED_MAP = {
    'NB01': _NB01_REQUIRED,
    'NB02': _NB02_REQUIRED,
    'NB03': _NB03_REQUIRED,
    'NB04': _NB04_REQUIRED,
}


def _validate_section(config: dict, section_name: str, subsections: dict) -> List[str]:
    """Validate a config section. Returns list of missing key paths."""
    missing = []
    if section_name not in config or not isinstance(config[section_name], dict):
        missing.append(section_name)
        return missing

    section = config[section_name]
    for subsection_name, keys in subsections.items():
        if not keys:
            if subsection_name not in section:
                missing.append(f"{section_name}.{subsection_name}")
            continue

        if subsection_name not in section or not isinstance(section[subsection_name], dict):
            missing.append(f"{section_name}.{subsection_name}")
            continue

        subsection = section[subsection_name]
        for key in keys:
            if key not in subsection:
                missing.append(f"{section_name}.{subsection_name}.{key}")

    return missing


def validate_notebook_config(config: dict, notebook: str) -> None:
    """Validate that config has all required keys for a given notebook.

    notebook: one of 'NB01', 'NB02', 'NB03', 'NB04'
    Raises ConfigError with all missing keys listed.
    """
    if config is None:
        raise ConfigError("Config is None — no configuration was loaded")
    if not isinstance(config, dict):
        raise ConfigError(f"Config must be dict, got {type(config).__name__}")
    if notebook not in _REQUIRED_MAP:
        raise ConfigError(f"Unknown notebook: {notebook}. Expected one of {list(_REQUIRED_MAP.keys())}")

    all_missing = []
    for section_name, subsections in _REQUIRED_MAP[notebook].items():
        all_missing.extend(_validate_section(config, section_name, subsections))

    if all_missing:
        raise ConfigError(
            f"Config validation failed for {notebook}. "
            f"Missing {len(all_missing)} required key(s):\n  " +
            "\n  ".join(all_missing)
        )

    print(f"Config validated for {notebook}: all required keys present")
