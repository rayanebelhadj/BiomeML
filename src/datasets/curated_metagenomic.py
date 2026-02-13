"""
curatedMetagenomicData dataset loader.

Curated collection of uniformly processed human microbiome data from
multiple studies.  Best dataset for disease prediction.

Reference: Pasolli et al. (2017) Nature Methods
https://www.nature.com/articles/nmeth.4468
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

from .base import BaseDataset


class CuratedMetagenomicDataset(BaseDataset):
    """Loader for curatedMetagenomicData."""

    STUDIES = {
        'ZellerG_2014':      {'disease': 'CRC',       'samples': 199},
        'YuJ_2015':          {'disease': 'CRC',       'samples': 128},
        'FengQ_2015':        {'disease': 'CRC',       'samples': 154},
        'VogtmannE_2016':    {'disease': 'CRC',       'samples': 104},
        'ThomasAM_2019':     {'disease': 'CRC',       'samples': 80},
        'NielsenHB_2014':    {'disease': 'IBD',       'samples': 396},
        'HallAB_2017':       {'disease': 'IBD',       'samples': 310},
        'LloydPriceJ_2019':  {'disease': 'IBD',       'samples': 1338},
        'QinJ_2012':         {'disease': 'T2D',       'samples': 368},
        'KarlssonFH_2013':   {'disease': 'T2D',       'samples': 145},
        'QinN_2014':         {'disease': 'Cirrhosis', 'samples': 237},
        'LeChatelierE_2013': {'disease': 'Obesity',   'samples': 292},
    }

    DISEASES = ['CRC', 'IBD', 'T2D', 'Cirrhosis', 'Obesity']

    DISEASE_MAPPINGS = {
        'CRC': {
            'case': ['crc', 'colorectal cancer', 'adenoma', 'carcinoma'],
            'control': ['control', 'healthy'],
        },
        'IBD': {
            'case': ['ibd', 'cd', 'uc'],
            'control': ['control', 'healthy'],
        },
        'T2D': {
            'case': ['t2d', 'type 2 diabetes'],
            'control': ['control', 'healthy', 'ngt'],
        },
        'Cirrhosis': {
            'case': ['cirrhosis'],
            'control': ['control', 'healthy'],
        },
        'Obesity': {
            'case': ['obese', 'obesity'],
            'control': ['control', 'lean', 'healthy'],
        },
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = 'cmd'
        self.data_dir = Path(config['data_dir']) / 'curatedMetagenomicData'
        self.studies = config.get('studies', list(self.STUDIES.keys()))
        self.abundance_path = self.data_dir / config['abundance_file']
        self.metadata_path = self.data_dir / config['metadata_file']
        self.tax_level = config.get('taxonomic_level', 'species')

    def download(self) -> None:
        print("curatedMetagenomicData Download Instructions")
        print("=" * 60)
        print()
        print("Option 1: Use R/Bioconductor")
        print("-" * 60)
        print('  BiocManager::install("curatedMetagenomicData")')
        print('  library(curatedMetagenomicData)')
        print()
        print("Option 2: Download from Zenodo")
        print("-" * 60)
        print("  Search for 'curatedMetagenomicData' on https://zenodo.org/")
        print()
        print("Available studies:")
        for study, info in self.STUDIES.items():
            print(f"  {study}: {info['disease']} ({info['samples']} samples)")
        print()
        print("Place files in:", self.data_dir)
        print(f"  - {self.abundance_path.name}")
        print(f"  - {self.metadata_path.name}")

    def load_abundance_data(self) -> pd.DataFrame:
        if self._abundance_data is not None:
            return self._abundance_data

        if not self.abundance_path.exists():
            alt_paths = [
                self.data_dir / 'relative_abundance.tsv',
                self.data_dir / 'taxonomic_profiles.tsv',
                self.data_dir / 'abundance.csv',
            ]
            for alt in alt_paths:
                if alt.exists():
                    self.abundance_path = alt
                    break
            else:
                raise FileNotFoundError(
                    f"Abundance file not found: {self.abundance_path}\n"
                    "Run dataset.download() for instructions."
                )

        sep = '\t' if self.abundance_path.suffix in ['.tsv', '.txt'] else ','
        df = pd.read_csv(self.abundance_path, sep=sep, index_col=0)

        first_idx = str(df.index[0]) if len(df.index) > 0 else ''
        if '|' in first_idx or '__' in first_idx or first_idx.startswith(('species:', 'genus:', 'family:')):
            df = df.T

        self._abundance_data = df
        return self._abundance_data

    def load_metadata(self) -> pd.DataFrame:
        if self._metadata is not None:
            return self._metadata

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}\n"
                "Run dataset.download() for instructions."
            )

        sep = '\t' if self.metadata_path.suffix in ['.tsv', '.txt'] else ','
        self._metadata = pd.read_csv(self.metadata_path, sep=sep, index_col=0)
        self._standardize_columns()

        if 'sample_id' in self._metadata.columns:
            self._metadata = self._metadata.drop_duplicates(subset='sample_id', keep='first')
            self._metadata = self._metadata.set_index('sample_id')

        return self._metadata

    def _standardize_columns(self):
        if self._metadata is None:
            return
        mappings = {
            'study_condition': 'disease_status',
            'disease': 'disease_status',
            'body_site': 'body_site',
            'age': 'age_years',
            'age_category': 'age_category',
            'gender': 'sex',
            'BMI': 'bmi',
            'study_name': 'study',
        }
        for old, new in mappings.items():
            if old in self._metadata.columns and new not in self._metadata.columns:
                self._metadata.rename(columns={old: new}, inplace=True)

    def load_phylogeny(self) -> Any:
        """CMD doesn't include phylogeny -- graph construction uses k-NN."""
        return None

    def get_disease_labels(self, disease_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if disease_name not in self.DISEASE_MAPPINGS:
            raise ValueError(
                f"Unknown disease: {disease_name}. "
                f"Available: {list(self.DISEASE_MAPPINGS.keys())}"
            )

        metadata = self.load_metadata()

        disease_col = None
        for col in ['disease_status', 'study_condition', 'disease']:
            if col in metadata.columns:
                disease_col = col
                break
        if disease_col is None:
            raise ValueError(
                f"No disease column found in metadata. "
                f"Columns: {list(metadata.columns)[:10]}"
            )

        mapping = self.DISEASE_MAPPINGS[disease_name]
        valid_mask = metadata[disease_col].notna()
        disease_values = metadata.loc[valid_mask, disease_col].astype(str).str.lower()

        is_case = pd.Series(False, index=disease_values.index)
        is_control = pd.Series(False, index=disease_values.index)

        for pattern in mapping['case']:
            is_case = is_case | disease_values.str.contains(pattern, na=False)
        for pattern in mapping['control']:
            is_control = is_control | (disease_values == pattern)

        is_control = is_control & ~is_case
        clear_status = is_case | is_control
        sample_ids = metadata.loc[valid_mask].index[clear_status].values
        labels = is_case[clear_status].astype(int).values

        if len(sample_ids) == 0:
            raise ValueError(f"No valid case/control samples found for {disease_name}")

        print(f"[CMD] {disease_name}: {sum(labels)} cases, {len(labels) - sum(labels)} controls")
        return sample_ids, labels

    def list_available_conditions(self) -> List[str]:
        return self.DISEASES.copy()

    def get_studies_for_disease(self, disease: str) -> List[str]:
        return [s for s, info in self.STUDIES.items() if info['disease'] == disease]

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

    def info(self) -> Dict[str, Any]:
        info = super().info()
        info.update({
            'available_studies': list(self.STUDIES.keys()),
            'diseases': self.DISEASES,
            'taxonomic_level': self.tax_level,
            'reference': 'Pasolli et al. 2017 Nature Methods',
        })
        return info
