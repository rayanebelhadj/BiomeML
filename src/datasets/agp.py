"""
American Gut Project dataset loader.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

from .base import BaseDataset


class AmericanGutDataset(BaseDataset):
    """Loader for American Gut Project data."""

    DISEASES = [
        'IBD', 'Diabetes', 'Cancer', 'Autoimmune', 'Depression',
        'Mental_Illness', 'PTSD', 'Arthritis', 'Asthma', 'Stomach_Bowel',
    ]

    DISEASE_COLUMNS = {
        'IBD': 'ibd',
        'Diabetes': 'diabetes',
        'Cancer': 'cancer',
        'Autoimmune': 'autoimmune',
        'Depression': 'depression',
        'Mental_Illness': 'mental_illness',
        'PTSD': 'ptsd',
        'Arthritis': 'arthritis',
        'Asthma': 'asthma',
        'Stomach_Bowel': 'stomach_bowel',
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = 'agp'
        self.data_dir = Path(config['data_dir']) / 'agp'
        self.biom_path = self.data_dir / config['biom_file']
        self.metadata_path = self.data_dir / config['metadata_file']
        self.phylogeny_path = self.data_dir / config['phylogeny_file']
        self.min_reads = config['min_reads_per_sample']
        self.min_prevalence = config['min_feature_prevalence']

    def download(self) -> None:
        print("American Gut Project data download instructions:")
        print("=" * 50)
        print("1. Go to https://qiita.ucsd.edu/")
        print("2. Search for 'American Gut Project' (Study ID: 10317)")
        print("3. Download the BIOM table and metadata")
        print("4. Place files in:", self.data_dir)
        print()
        print("Expected files:")
        print(f"  - {self.biom_path.name} (BIOM abundance table)")
        print(f"  - {self.metadata_path.name} (Sample metadata)")
        print(f"  - {self.phylogeny_path.name} (Phylogenetic tree)")

    def load_abundance_data(self) -> Any:
        if self._abundance_data is not None:
            return self._abundance_data
        if not self.biom_path.exists():
            raise FileNotFoundError(f"BIOM file not found: {self.biom_path}")
        from biom import load_table
        self._abundance_data = load_table(str(self.biom_path))
        return self._abundance_data

    def load_metadata(self) -> pd.DataFrame:
        if self._metadata is not None:
            return self._metadata
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        self._metadata = pd.read_csv(
            self.metadata_path, sep='\t', index_col=0, low_memory=False,
        )
        return self._metadata

    def load_phylogeny(self) -> Any:
        if self._phylogeny is not None:
            return self._phylogeny
        if not self.phylogeny_path.exists():
            raise FileNotFoundError(f"Phylogeny file not found: {self.phylogeny_path}")
        from Bio import Phylo
        self._phylogeny = Phylo.read(str(self.phylogeny_path), 'newick')
        return self._phylogeny

    def get_disease_labels(self, disease_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if disease_name not in self.DISEASES:
            raise ValueError(f"Unknown disease: {disease_name}. Supported: {self.DISEASES}")

        metadata = self.load_metadata()
        col_name = self.DISEASE_COLUMNS[disease_name]

        if col_name not in metadata.columns:
            col_matches = [c for c in metadata.columns if c.lower() == col_name.lower()]
            if col_matches:
                col_name = col_matches[0]
            else:
                raise ValueError(f"Column '{col_name}' not found in metadata")

        valid_mask = metadata[col_name].notna()
        disease_col = metadata.loc[valid_mask, col_name].astype(str).str.lower()

        is_case = (
            disease_col.str.contains('diagnosed', na=False)
            | disease_col.str.contains('yes', na=False)
            | (disease_col == 'true')
            | (disease_col == '1')
        )
        is_control = (
            disease_col.str.contains('i do not have this condition', na=False)
            | disease_col.str.contains('no', na=False)
            | (disease_col == 'false')
            | (disease_col == '0')
        )
        is_control = is_control & ~is_case

        clear_status = is_case | is_control
        sample_ids = metadata.loc[valid_mask].index[clear_status].values
        labels = is_case[clear_status].astype(int).values

        if len(sample_ids) == 0:
            raise ValueError(f"No valid case/control samples found for {disease_name}")

        return sample_ids, labels

    def list_available_conditions(self) -> List[str]:
        return self.DISEASES.copy()

    def get_abundance_for_samples(
        self, sample_ids: List[str], normalize: bool = True,
    ) -> pd.DataFrame:
        biom_table = self.load_abundance_data()
        available_samples = set(biom_table.ids(axis='sample'))
        valid_samples = [s for s in sample_ids if s in available_samples]

        if not valid_samples:
            raise ValueError("No requested samples found in BIOM table")

        filtered_biom = biom_table.filter(valid_samples, axis='sample', inplace=False)
        df = filtered_biom.to_dataframe().T

        if normalize:
            row_sums = df.sum(axis=1)
            row_sums = row_sums.replace(0, 1)
            df = df.div(row_sums, axis=0)

        return df

    def info(self) -> Dict[str, Any]:
        info = super().info()
        info.update({
            'biom_path': str(self.biom_path),
            'metadata_path': str(self.metadata_path),
            'phylogeny_path': str(self.phylogeny_path),
            'diseases': self.DISEASES,
        })
        return info
