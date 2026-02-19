"""
Config-driven dataset loader.

Reads everything from YAML: file paths, column mappings, condition
definitions.  No dataset-specific Python code needed -- just a YAML file.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np

from .base import BaseDataset

# Standard column names used throughout the pipeline
STANDARD_COLUMNS = [
    'sample_id', 'age', 'sex', 'bmi', 'body_site', 'study',
    'disease_status', 'antibiotics',
]


class ConfigDrivenDataset(BaseDataset):
    """
    Fully YAML-driven dataset loader.

    All dataset-specific knowledge (file paths, column names, condition
    definitions) comes from the config dict, which is loaded from a YAML
    file in datasets_config/.

    Required config keys:
        dataset_name, data_dir, abundance_file, metadata_file

    Optional config keys:
        phylogeny_file, columns, conditions, sample_id_column
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = config['dataset_name']
        self.data_dir = self.data_dir / self.name

        self.abundance_path = self.data_dir / config['abundance_file']
        self.metadata_path = self.data_dir / config['metadata_file']

        phylo_file = config.get('phylogeny_file')
        self.phylogeny_path = self.data_dir / phylo_file if phylo_file else None

        self.column_map: Dict[str, Optional[str]] = config.get('columns', {})
        self.conditions: Dict[str, Dict] = config.get('conditions', {})
        self.sample_id_col: Optional[str] = self.column_map.get('sample_id')

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def download(self) -> None:
        print(f"Dataset: {self.name}")
        print("=" * 50)
        print(f"Place your data files in: {self.data_dir}")
        print()
        print("Expected files:")
        print(f"  - {self.abundance_path.name}  (abundance table: TSV, CSV, or BIOM)")
        print(f"  - {self.metadata_path.name}  (sample metadata: TSV or CSV)")
        if self.phylogeny_path:
            print(f"  - {self.phylogeny_path.name}  (phylogenetic tree: Newick)")

    def load_abundance_data(self) -> pd.DataFrame:
        if self._abundance_data is not None:
            return self._abundance_data
        if not self.abundance_path.exists():
            raise FileNotFoundError(f"Abundance file not found: {self.abundance_path}")

        if self.abundance_path.suffix in ('.biom', '.biom.gz'):
            from biom import load_table
            biom_table = load_table(str(self.abundance_path))
            self._abundance_data = biom_table.to_dataframe().T
        else:
            sep = _detect_separator(self.abundance_path)
            df = pd.read_csv(self.abundance_path, sep=sep, index_col=0)
            if _looks_like_features_in_rows(df):
                df = df.T
            self._abundance_data = df

        return self._abundance_data

    def load_metadata(self) -> pd.DataFrame:
        if self._metadata is not None:
            return self._metadata
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        sep = _detect_separator(self.metadata_path)
        if self.sample_id_col:
            self._metadata = pd.read_csv(self.metadata_path, sep=sep, low_memory=False)
            if self.sample_id_col in self._metadata.columns:
                self._metadata.set_index(self.sample_id_col, inplace=True)
        else:
            self._metadata = pd.read_csv(self.metadata_path, sep=sep, index_col=0, low_memory=False)

        self._standardize_columns()
        return self._metadata

    def load_phylogeny(self) -> Any:
        if self._phylogeny is not None:
            return self._phylogeny
        if self.phylogeny_path is None or not self.phylogeny_path.exists():
            return None
        from Bio import Phylo
        self._phylogeny = Phylo.read(str(self.phylogeny_path), 'newick')
        return self._phylogeny

    # ------------------------------------------------------------------
    # Column standardization
    # ------------------------------------------------------------------

    def _standardize_columns(self):
        """Rename metadata columns to standard names based on self.column_map."""
        if self._metadata is None or not self.column_map:
            return
        rename = {}
        for standard_name, source_col in self.column_map.items():
            if standard_name == 'sample_id':
                continue
            if source_col and source_col in self._metadata.columns:
                if standard_name not in self._metadata.columns:
                    rename[source_col] = standard_name
        if rename:
            self._metadata.rename(columns=rename, inplace=True)

    def get_standardized_column(self, standard_name: str) -> Optional[str]:
        """Return the actual column name in metadata for a standard name.

        Checks the column_map first, then falls back to checking if the
        standard name itself exists in metadata.
        """
        mapped = self.column_map.get(standard_name)
        if mapped:
            return mapped
        if self._metadata is not None and standard_name in self._metadata.columns:
            return standard_name
        return None

    # ------------------------------------------------------------------
    # Conditions / labels
    # ------------------------------------------------------------------

    def get_disease_labels(self, disease_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if disease_name not in self.conditions:
            raise ValueError(
                f"Unknown condition: '{disease_name}'. "
                f"Available: {self.list_available_conditions()}"
            )

        cond = self.conditions[disease_name]
        column = cond['column']
        case_vals = [str(v).lower() for v in cond['case_values']]
        control_vals = [str(v).lower() for v in cond['control_values']]

        metadata = self.load_metadata()

        if column not in metadata.columns:
            raise ValueError(
                f"Column '{column}' (for condition '{disease_name}') "
                f"not found in metadata. Available: {list(metadata.columns)[:15]}"
            )

        valid_mask = metadata[column].notna()
        values = metadata.loc[valid_mask, column].astype(str).str.lower().str.strip()

        is_case = values.isin(case_vals)
        is_control = values.isin(control_vals)

        for pattern in case_vals:
            is_case = is_case | values.str.contains(pattern, na=False)
        is_control = is_control & ~is_case

        clear = is_case | is_control
        sample_ids = metadata.loc[valid_mask].index[clear].values
        labels = is_case[clear].astype(int).values

        if len(sample_ids) == 0:
            raise ValueError(
                f"No valid case/control samples for '{disease_name}'. "
                f"Column='{column}', case_values={case_vals}, "
                f"control_values={control_vals}, "
                f"unique values found: {list(values.unique()[:10])}"
            )

        n_case = int(labels.sum())
        n_ctrl = len(labels) - n_case
        print(f"[{self.name}] {disease_name}: {n_case} cases, {n_ctrl} controls")
        return sample_ids, labels

    def list_available_conditions(self) -> List[str]:
        return list(self.conditions.keys())

    # ------------------------------------------------------------------
    # Abundance
    # ------------------------------------------------------------------

    def get_abundance_for_samples(
        self, sample_ids: List[str], normalize: bool = True,
    ) -> pd.DataFrame:
        abundance = self.load_abundance_data()
        str_ids = [str(s) for s in sample_ids]

        if set(str_ids) & set(str(i) for i in abundance.index):
            idx_map = {str(i): i for i in abundance.index}
            valid = [idx_map[s] for s in str_ids if s in idx_map]
            df = abundance.loc[valid]
        elif set(str_ids) & set(str(c) for c in abundance.columns):
            col_map = {str(c): c for c in abundance.columns}
            valid = [col_map[s] for s in str_ids if s in col_map]
            df = abundance[valid].T
        else:
            raise ValueError("No requested samples found in abundance data")

        if normalize:
            row_sums = df.sum(axis=1).replace(0, 1)
            df = df.div(row_sums, axis=0)

        return df

    # ------------------------------------------------------------------
    # Validation & info
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        super().validate()
        abundance = self.load_abundance_data()
        metadata = self.load_metadata()
        a_ids = set(str(i) for i in abundance.index) | set(str(c) for c in abundance.columns)
        m_ids = set(str(i) for i in metadata.index)
        if not (a_ids & m_ids):
            raise ValueError(
                "No sample ID overlap between abundance and metadata. "
                f"Abundance IDs (first 5): {list(a_ids)[:5]}, "
                f"Metadata IDs (first 5): {list(m_ids)[:5]}"
            )
        return True

    def info(self) -> Dict[str, Any]:
        info = super().info()
        info.update({
            'abundance_path': str(self.abundance_path),
            'metadata_path': str(self.metadata_path),
            'phylogeny_available': self.phylogeny_path is not None and self.phylogeny_path.exists(),
            'column_mappings': self.column_map,
        })
        return info


# ======================================================================
# Helpers
# ======================================================================

def _detect_separator(path: Path) -> str:
    """Auto-detect CSV vs TSV from extension or first line."""
    if path.suffix in ('.tsv', '.txt'):
        return '\t'
    if path.suffix == '.csv':
        return ','
    with open(path) as f:
        first_line = f.readline()
    if '\t' in first_line:
        return '\t'
    return ','


def _looks_like_features_in_rows(df: pd.DataFrame) -> bool:
    """Heuristic: if index looks like taxonomy strings, transpose."""
    if len(df.index) == 0:
        return False
    first_idx = str(df.index[0])
    return ('|' in first_idx or '__' in first_idx
            or first_idx.startswith(('k__', 'p__', 'species:', 'genus:')))
