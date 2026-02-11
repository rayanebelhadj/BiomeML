"""
Human Microbiome Project dataset loader.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

from .base import BaseDataset


class HumanMicrobiomeDataset(BaseDataset):
    """Loader for Human Microbiome Project data (HMP2 / IBDMDB)."""

    BODY_SITES = ['Gut', 'Oral', 'Skin', 'Vaginal', 'Nasal']

    CONDITIONS = ['IBD', 'Crohns', 'UC', 'Body_Site_Classification']

    DATA_SOURCES = {
        'hmp1': 'Human Microbiome Project Phase 1 (healthy adults)',
        'hmp2': 'Human Microbiome Project Phase 2 (IBD focus)',
        'ibdmdb': 'Inflammatory Bowel Disease Multi-omics Database',
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = 'hmp'
        self.data_dir = Path(config['data_dir']) / 'hmp'
        self.source = config.get('hmp_source', 'hmp2')
        self.biom_path = self.data_dir / config['biom_file']
        self.metadata_path = self.data_dir / config['metadata_file']
        self.phylogeny_path = self.data_dir / config.get('phylogeny_file', 'phylogeny.nwk')
        self.min_reads = config['min_reads_per_sample']
        self.body_site = config.get('body_site', 'Gut')

    def download(self) -> None:
        print("Human Microbiome Project data download instructions:")
        print("=" * 50)
        print()
        print("Option 1: HMP2 / IBDMDB (recommended for disease prediction)")
        print("  1. Go to https://ibdmdb.org/")
        print("  2. Navigate to Data > Downloads")
        print("  3. Download taxonomic profiles + metadata")
        print("  4. Place files in:", self.data_dir)
        print()
        print("Option 2: HMP DACC — https://www.hmpdacc.org/")
        print()
        print("Expected files:")
        print(f"  - {self.biom_path.name}")
        print(f"  - {self.metadata_path.name}")

    def load_abundance_data(self) -> Any:
        if self._abundance_data is not None:
            return self._abundance_data

        if not self.biom_path.exists():
            for ext in ['.tsv', '.csv', '.txt']:
                alt_path = self.biom_path.with_suffix(ext)
                if alt_path.exists():
                    sep = '\t' if ext in ['.tsv', '.txt'] else ','
                    self._abundance_data = pd.read_csv(alt_path, sep=sep, index_col=0)
                    return self._abundance_data
            raise FileNotFoundError(
                f"BIOM file not found: {self.biom_path}\n"
                "Run dataset.download() for instructions."
            )

        from biom import load_table
        self._abundance_data = load_table(str(self.biom_path))
        return self._abundance_data

    def load_metadata(self) -> pd.DataFrame:
        if self._metadata is not None:
            return self._metadata
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}\n"
                "Run dataset.download() for instructions."
            )
        self._metadata = pd.read_csv(
            self.metadata_path, sep='\t', index_col=0, low_memory=False,
        )
        self._standardize_metadata_columns()
        return self._metadata

    def _standardize_metadata_columns(self):
        if self._metadata is None:
            return
        column_mappings = {
            'body_site': 'body_site', 'body_habitat': 'body_site',
            'sample_body_site': 'body_site', 'hmp_body_site': 'body_site',
            'diagnosis': 'disease_status', 'disease': 'disease_status',
            'ibd_diagnosis': 'disease_status', 'health_status': 'disease_status',
            'subject_id': 'subject_id', 'participant_id': 'subject_id',
            'host_subject_id': 'subject_id',
            'age': 'age_years', 'age_at_diagnosis': 'age_years', 'host_age': 'age_years',
            'sex': 'sex', 'gender': 'sex', 'host_sex': 'sex',
        }
        for old_name, new_name in column_mappings.items():
            for col in self._metadata.columns:
                if col.lower() == old_name.lower() and new_name not in self._metadata.columns:
                    self._metadata.rename(columns={col: new_name}, inplace=True)
                    break

    def load_phylogeny(self) -> Any:
        if self._phylogeny is not None:
            return self._phylogeny
        if not self.phylogeny_path.exists():
            return None
        from Bio import Phylo
        self._phylogeny = Phylo.read(str(self.phylogeny_path), 'newick')
        return self._phylogeny

    def get_disease_labels(self, disease_name: str) -> Tuple[np.ndarray, np.ndarray]:
        metadata = self.load_metadata()

        if disease_name == 'IBD':
            return self._get_ibd_labels(metadata)
        elif disease_name in ('Crohns', 'UC'):
            return self._get_subtype_labels(metadata, disease_name)
        elif disease_name == 'Body_Site_Classification':
            return self._get_body_site_labels(metadata)
        else:
            raise ValueError(f"Unknown condition: {disease_name}. Supported: {self.CONDITIONS}")

    def _get_ibd_labels(self, metadata: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        disease_col = self._find_disease_col(metadata)
        valid_mask = metadata[disease_col].notna()
        values = metadata.loc[valid_mask, disease_col].astype(str).str.lower()

        case_values = ['cd', 'uc', 'ibd', "crohn's disease", 'ulcerative colitis']
        control_values = ['nonibd', 'non-ibd', 'healthy', 'control', 'hc']

        is_case = values.isin(case_values)
        is_control = values.isin(control_values)
        clear = is_case | is_control
        sample_ids = metadata.loc[valid_mask].index[clear].values
        labels = is_case[clear].astype(int).values

        if len(sample_ids) == 0:
            raise ValueError("No valid IBD case/control samples found")
        return sample_ids, labels

    def _get_subtype_labels(self, metadata: pd.DataFrame, subtype: str) -> Tuple[np.ndarray, np.ndarray]:
        disease_col = self._find_disease_col(metadata)
        valid_mask = metadata[disease_col].notna()
        values = metadata.loc[valid_mask, disease_col].astype(str).str.lower()

        case_val = 'cd' if subtype == 'Crohns' else 'uc'
        is_case = values == case_val
        is_control = values.isin(['nonibd', 'non-ibd', 'healthy', 'control', 'hc'])
        clear = is_case | is_control
        sample_ids = metadata.loc[valid_mask].index[clear].values
        labels = is_case[clear].astype(int).values

        if len(sample_ids) == 0:
            raise ValueError(f"No valid {subtype} case/control samples found")
        return sample_ids, labels

    def _get_body_site_labels(self, metadata: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if 'body_site' not in metadata.columns:
            raise ValueError("No body_site column found in metadata")
        valid_mask = metadata['body_site'].notna()
        sample_ids = metadata.loc[valid_mask].index.values
        body_sites = metadata.loc[valid_mask, 'body_site'].values
        unique_sites = sorted(set(body_sites))
        site_to_int = {site: i for i, site in enumerate(unique_sites)}
        labels = np.array([site_to_int[site] for site in body_sites])
        return sample_ids, labels

    def _find_disease_col(self, metadata: pd.DataFrame) -> str:
        for col in ['disease_status', 'diagnosis', 'ibd_diagnosis', 'disease']:
            if col in metadata.columns:
                return col
        raise ValueError(
            f"No disease column found. Available: {list(metadata.columns)[:10]}..."
        )

    def list_available_conditions(self) -> List[str]:
        return self.CONDITIONS.copy()

    def get_abundance_for_samples(
        self, sample_ids: List[str], normalize: bool = True,
    ) -> pd.DataFrame:
        abundance = self.load_abundance_data()

        if hasattr(abundance, 'ids'):
            available = set(abundance.ids(axis='sample'))
            valid = [s for s in sample_ids if s in available]
            if not valid:
                raise ValueError("No requested samples found in abundance data")
            filtered = abundance.filter(valid, axis='sample', inplace=False)
            df = filtered.to_dataframe().T
        else:
            valid = [s for s in sample_ids if s in abundance.index]
            if not valid:
                valid = [s for s in sample_ids if s in abundance.columns]
                if not valid:
                    raise ValueError("No requested samples found in abundance data")
                df = abundance[valid].T
            else:
                df = abundance.loc[valid]

        if normalize:
            row_sums = df.sum(axis=1).replace(0, 1)
            df = df.div(row_sums, axis=0)

        return df

    def info(self) -> Dict[str, Any]:
        info = super().info()
        info.update({
            'source': self.source,
            'body_sites': self.BODY_SITES,
            'conditions': self.CONDITIONS,
            'data_sources': self.DATA_SOURCES,
        })
        return info
