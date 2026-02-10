"""
Base dataset interface for BiomeML.
All dataset loaders must inherit from BaseDataset.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


class BaseDataset(ABC):
    """Abstract base class for microbiome datasets."""

    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, dict):
            raise TypeError(f"config must be a dict, got {type(config).__name__}")
        if not config:
            raise ValueError("config must not be empty")

        self.config = config
        self.data_dir = Path(config['data_dir'])
        self.name = config['dataset_name']

        self._abundance_data = None
        self._metadata = None
        self._phylogeny = None

    @abstractmethod
    def download(self) -> None:
        """Download dataset files if not present."""
        pass

    @abstractmethod
    def load_abundance_data(self) -> Any:
        """Load OTU/ASV abundance table."""
        pass

    @abstractmethod
    def load_metadata(self) -> pd.DataFrame:
        """Load sample metadata."""
        pass

    @abstractmethod
    def load_phylogeny(self) -> Any:
        """Load phylogenetic tree. Return None if unavailable for this dataset."""
        pass

    @abstractmethod
    def get_disease_labels(self, disease_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract disease labels from metadata.

        Returns:
            Tuple of (sample_ids, labels) where labels are 0 (control) or 1 (case)
        """
        pass

    @abstractmethod
    def list_available_conditions(self) -> List[str]:
        """List available diseases/conditions in the dataset."""
        pass

    @abstractmethod
    def get_abundance_for_samples(
        self,
        sample_ids: List[str],
        normalize: bool = True,
    ) -> pd.DataFrame:
        """Get abundance data (samples x features) for specific sample IDs."""
        pass

    def validate(self) -> bool:
        """Validate dataset integrity. Raises on failure."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        self.load_abundance_data()
        self.load_metadata()
        return True

    def get_sample_count(self) -> int:
        """Get total number of samples."""
        if self._metadata is None:
            self.load_metadata()
        return len(self._metadata)

    def get_feature_count(self) -> int:
        """Get total number of features (OTUs/ASVs)."""
        if self._abundance_data is None:
            self.load_abundance_data()
        if hasattr(self._abundance_data, 'shape'):
            return self._abundance_data.shape[1]
        return 0

    def info(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'data_dir': str(self.data_dir),
            'conditions': self.list_available_conditions(),
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', data_dir='{self.data_dir}')"
