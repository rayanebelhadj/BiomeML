"""
Custom dataset loader for user-provided data.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

from .base import BaseDataset


class CustomDataset(BaseDataset):
    """
    Loader for custom user-provided microbiome data.

    Expects:
    - Abundance table (BIOM or TSV format)
    - Metadata file (TSV format)
    - Optional phylogenetic tree (Newick format)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = config['dataset_name']
        self.data_dir = Path(config['data_dir']) / 'custom'
        self.abundance_path = self.data_dir / config['abundance_file']
        self.metadata_path = self.data_dir / config['metadata_file']
        self.phylogeny_path = self.data_dir / config.get('phylogeny_file', 'phylogeny.nwk')
        self.sample_id_col = config.get('sample_id_column', None)
        self.target_col = config['target_column']
        self.case_values = config['case_values']
        self.control_values = config['control_values']

    def download(self) -> None:
        print("Custom Dataset Setup Instructions:")
        print("=" * 50)
        print()
        print("Place your data files in:", self.data_dir)
        print()
        print("Expected files:")
        print(f"  1. {self.abundance_path.name} — TSV or BIOM")
        print(f"  2. {self.metadata_path.name} — TSV with target column")
        print(f"  3. {self.phylogeny_path.name} (optional) — Newick")

    def load_abundance_data(self) -> pd.DataFrame:
        if self._abundance_data is not None:
            return self._abundance_data
        if not self.abundance_path.exists():
            raise FileNotFoundError(f"Abundance file not found: {self.abundance_path}")

        if self.abundance_path.suffix in ['.biom', '.biom.gz']:
            from biom import load_table
            biom_table = load_table(str(self.abundance_path))
            self._abundance_data = biom_table.to_dataframe().T
        else:
            sep = '\t' if self.abundance_path.suffix in ['.tsv', '.txt'] else ','
            self._abundance_data = pd.read_csv(self.abundance_path, sep=sep, index_col=0)

        return self._abundance_data

    def load_metadata(self) -> pd.DataFrame:
        if self._metadata is not None:
            return self._metadata
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        sep = '\t' if self.metadata_path.suffix in ['.tsv', '.txt'] else ','
        if self.sample_id_col:
            self._metadata = pd.read_csv(self.metadata_path, sep=sep)
            self._metadata.set_index(self.sample_id_col, inplace=True)
        else:
            self._metadata = pd.read_csv(self.metadata_path, sep=sep, index_col=0)
        return self._metadata

    def load_phylogeny(self) -> Any:
        if self._phylogeny is not None:
            return self._phylogeny
        if not self.phylogeny_path.exists():
            return None
        from Bio import Phylo
        self._phylogeny = Phylo.read(str(self.phylogeny_path), 'newick')
        return self._phylogeny

    def get_disease_labels(self, disease_name: str = None) -> Tuple[np.ndarray, np.ndarray]:
        metadata = self.load_metadata()
        target_col = disease_name if (disease_name and disease_name in metadata.columns) else self.target_col

        if target_col not in metadata.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in metadata. "
                f"Available: {list(metadata.columns)}"
            )

        valid_mask = metadata[target_col].notna()
        target_values = metadata.loc[valid_mask, target_col].astype(str).str.lower().str.strip()

        case_vals = [v.lower() for v in self.case_values]
        control_vals = [v.lower() for v in self.control_values]

        is_case = target_values.isin(case_vals)
        is_control = target_values.isin(control_vals)
        clear_status = is_case | is_control

        if clear_status.sum() == 0:
            try:
                numeric_vals = pd.to_numeric(target_values, errors='coerce')
                is_case = numeric_vals == 1
                is_control = numeric_vals == 0
                clear_status = is_case | is_control
            except Exception:
                pass

        if clear_status.sum() == 0:
            raise ValueError(
                f"No valid case/control values found in '{target_col}'. "
                f"Found: {target_values.unique()[:10]}. "
                f"Expected case={self.case_values}, control={self.control_values}"
            )

        sample_ids = metadata.loc[valid_mask].index[clear_status].values
        labels = is_case[clear_status].astype(int).values
        return sample_ids, labels

    def list_available_conditions(self) -> List[str]:
        return [self.target_col]

    def get_abundance_for_samples(
        self, sample_ids: List[str], normalize: bool = True,
    ) -> pd.DataFrame:
        abundance = self.load_abundance_data()

        if set(sample_ids) & set(abundance.index):
            df = abundance.loc[[s for s in sample_ids if s in abundance.index]]
        elif set(sample_ids) & set(abundance.columns):
            df = abundance[[s for s in sample_ids if s in abundance.columns]].T
        else:
            raise ValueError("No requested samples found in abundance data")

        if normalize:
            row_sums = df.sum(axis=1).replace(0, 1)
            df = df.div(row_sums, axis=0)
        return df

    def validate(self) -> bool:
        super().validate()
        abundance = self.load_abundance_data()
        metadata = self.load_metadata()
        abundance_samples = set(abundance.index) | set(abundance.columns)
        metadata_samples = set(metadata.index)
        overlap = abundance_samples & metadata_samples
        if not overlap:
            raise ValueError("No sample overlap between abundance and metadata.")
        return True

    def info(self) -> Dict[str, Any]:
        info = super().info()
        info.update({
            'abundance_path': str(self.abundance_path),
            'metadata_path': str(self.metadata_path),
            'target_column': self.target_col,
        })
        return info
