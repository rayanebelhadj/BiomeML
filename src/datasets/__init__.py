"""
Dataset registry for BiomeML.
Unified interface for loading microbiome datasets.

Loading order:
  1. If config has ``loader: "agp"`` -> use the specialized class
  2. If dataset_name matches a known loader -> use that class
  3. Otherwise -> use ConfigDrivenDataset (reads everything from YAML)
"""

from typing import Dict, Any, List

from .base import BaseDataset
from .config_driven import ConfigDrivenDataset
from .agp import AmericanGutDataset
from .hmp import HumanMicrobiomeDataset
from .curated_metagenomic import CuratedMetagenomicDataset

# Keep old CustomDataset importable for back-compat
from .custom import CustomDataset


SPECIALIZED_LOADERS = {
    'agp': AmericanGutDataset,
    'american_gut': AmericanGutDataset,
    'hmp': HumanMicrobiomeDataset,
    'human_microbiome': HumanMicrobiomeDataset,
    'cmd': CuratedMetagenomicDataset,
    'curatedmetagenomicdata': CuratedMetagenomicDataset,
    'curated': CuratedMetagenomicDataset,
}

# Kept for back-compat; includes 'custom' key
DATASET_REGISTRY = {
    **SPECIALIZED_LOADERS,
    'custom': CustomDataset,
}


def load_dataset(dataset_name: str, config: Dict[str, Any]) -> BaseDataset:
    """Load a dataset.

    Resolution order:
      1. ``config['loader']`` — explicit specialized loader name
      2. ``dataset_name`` matches a specialized loader (agp, cmd, hmp)
      3. Fall back to ConfigDrivenDataset (generic, YAML-driven)

    This means any new dataset works out of the box with just a YAML
    file — no Python code required.
    """
    explicit_loader = config.get('loader', '').lower()
    if explicit_loader and explicit_loader in SPECIALIZED_LOADERS:
        return SPECIALIZED_LOADERS[explicit_loader](config)

    name_lower = dataset_name.lower()
    if name_lower in SPECIALIZED_LOADERS:
        return SPECIALIZED_LOADERS[name_lower](config)

    if 'conditions' not in config:
        raise ValueError(
            f"Dataset '{dataset_name}' has no specialized loader and no "
            f"'conditions' section in config. Either use a known dataset name "
            f"({list(SPECIALIZED_LOADERS.keys())}) or define conditions in "
            f"your dataset YAML."
        )

    return ConfigDrivenDataset(config)


def list_datasets() -> List[str]:
    """Return canonical dataset names (specialized + generic)."""
    return ['agp', 'hmp', 'cmd', 'custom / config-driven']


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get static information about a dataset without loading data."""
    info = {
        'agp': {
            'name': 'American Gut Project',
            'description': 'Large-scale citizen science microbiome project',
            'conditions': AmericanGutDataset.DISEASES,
            'source': 'https://americangut.org/',
            'format': 'BIOM + TSV metadata',
        },
        'hmp': {
            'name': 'Human Microbiome Project',
            'description': 'NIH-funded characterization of human microbiome',
            'conditions': HumanMicrobiomeDataset.CONDITIONS,
            'body_sites': HumanMicrobiomeDataset.BODY_SITES,
            'source': 'https://hmpdacc.org/',
            'format': 'BIOM + TSV metadata',
        },
        'cmd': {
            'name': 'curatedMetagenomicData',
            'description': 'Curated uniformly processed human microbiome data (best for ML)',
            'conditions': CuratedMetagenomicDataset.DISEASES,
            'studies': list(CuratedMetagenomicDataset.STUDIES.keys()),
            'source': 'https://waldronlab.io/curatedMetagenomicData/',
            'format': 'TSV (pre-processed)',
            'reference': 'Pasolli et al. 2017 Nature Methods',
        },
        'custom': {
            'name': 'Config-Driven Dataset',
            'description': 'Any dataset defined entirely via YAML config',
            'conditions': ['defined in YAML'],
            'format': 'TSV/CSV/BIOM + TSV/CSV metadata',
        },
    }
    name_lower = dataset_name.lower()
    aliases = {
        'american_gut': 'agp', 'human_microbiome': 'hmp',
        'curatedmetagenomicdata': 'cmd', 'curated': 'cmd',
        'config_driven': 'custom', 'generic': 'custom',
    }
    key = aliases.get(name_lower, name_lower)
    if key not in info:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return info[key]


__all__ = [
    'BaseDataset',
    'ConfigDrivenDataset',
    'AmericanGutDataset',
    'HumanMicrobiomeDataset',
    'CustomDataset',
    'CuratedMetagenomicDataset',
    'SPECIALIZED_LOADERS',
    'DATASET_REGISTRY',
    'load_dataset',
    'list_datasets',
    'get_dataset_info',
]
