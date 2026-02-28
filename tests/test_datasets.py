"""Tests for src/datasets/ module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile, os

from src.datasets.base import BaseDataset
from src.datasets.agp import AmericanGutDataset
from src.datasets.curated_metagenomic import CuratedMetagenomicDataset
from src.datasets.hmp import HumanMicrobiomeDataset
from src.datasets.custom import CustomDataset
from src.datasets.config_driven import ConfigDrivenDataset, _detect_separator
from src.datasets import (
    load_dataset, list_datasets, get_dataset_info,
    DATASET_REGISTRY, SPECIALIZED_LOADERS,
)


# ---------------------------------------------------------------------------
# BaseDataset
# ---------------------------------------------------------------------------

class TestBaseDataset:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseDataset({"data_dir": "data", "dataset_name": "test"})

    def test_rejects_non_dict_config(self):
        class Dummy(BaseDataset):
            def download(self): pass
            def load_abundance_data(self): pass
            def load_metadata(self): pass
            def load_phylogeny(self): pass
            def get_disease_labels(self, d): pass
            def list_available_conditions(self): pass
            def get_abundance_for_samples(self, s, n=True): pass

        with pytest.raises(TypeError):
            Dummy("not a dict")

    def test_rejects_empty_config(self):
        class Dummy(BaseDataset):
            def download(self): pass
            def load_abundance_data(self): pass
            def load_metadata(self): pass
            def load_phylogeny(self): pass
            def get_disease_labels(self, d): pass
            def list_available_conditions(self): pass
            def get_abundance_for_samples(self, s, n=True): pass

        with pytest.raises(ValueError):
            Dummy({})

    def test_missing_data_dir_raises(self):
        class Dummy(BaseDataset):
            def download(self): pass
            def load_abundance_data(self): pass
            def load_metadata(self): pass
            def load_phylogeny(self): pass
            def get_disease_labels(self, d): pass
            def list_available_conditions(self): pass
            def get_abundance_for_samples(self, s, n=True): pass

        with pytest.raises(KeyError):
            Dummy({"dataset_name": "x"})


# ---------------------------------------------------------------------------
# Registry & load_dataset
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_list_datasets(self):
        ds_list = list_datasets()
        assert "agp" in ds_list
        assert "cmd" in ds_list
        assert "hmp" in ds_list

    def test_load_unknown_without_conditions_raises(self):
        with pytest.raises(ValueError, match="no specialized loader"):
            load_dataset("nonexistent", {"data_dir": "data", "dataset_name": "x"})

    def test_registry_aliases(self):
        assert DATASET_REGISTRY["agp"] is DATASET_REGISTRY["american_gut"]
        assert DATASET_REGISTRY["cmd"] is DATASET_REGISTRY["curated"]

    def test_get_dataset_info(self):
        info = get_dataset_info("cmd")
        assert info["name"] == "curatedMetagenomicData"
        assert "CRC" in info["conditions"]

    def test_get_info_unknown_raises(self):
        with pytest.raises(ValueError):
            get_dataset_info("nonexistent")

    def test_explicit_loader_cmd_overrides_name(self):
        """loader: 'cmd' in config -> CuratedMetagenomicDataset even if name differs."""
        ds = load_dataset("my_custom_name", {
            "data_dir": "data", "dataset_name": "my_custom_name",
            "loader": "cmd",
            "abundance_file": "x.tsv", "metadata_file": "x.tsv",
        })
        assert isinstance(ds, CuratedMetagenomicDataset)

    def test_explicit_loader_agp(self):
        ds = load_dataset("whatever", {
            "data_dir": "data", "dataset_name": "whatever",
            "loader": "agp",
            "biom_file": "x.biom", "metadata_file": "x.tsv",
            "phylogeny_file": "x.nwk", "min_reads_per_sample": 1000,
            "min_feature_prevalence": 0.001,
        })
        assert isinstance(ds, AmericanGutDataset)

    def test_explicit_loader_hmp(self):
        ds = load_dataset("whatever", {
            "data_dir": "data", "dataset_name": "whatever",
            "loader": "hmp",
            "biom_file": "x.biom", "metadata_file": "x.tsv",
            "min_reads_per_sample": 1000,
        })
        assert isinstance(ds, HumanMicrobiomeDataset)

    def test_name_alias_american_gut(self):
        ds = load_dataset("american_gut", {
            "data_dir": "data", "dataset_name": "agp",
            "biom_file": "x.biom", "metadata_file": "x.tsv",
            "phylogeny_file": "x.nwk", "min_reads_per_sample": 1000,
            "min_feature_prevalence": 0.001,
        })
        assert isinstance(ds, AmericanGutDataset)

    def test_name_alias_curatedmetagenomicdata(self):
        ds = load_dataset("curatedmetagenomicdata", {
            "data_dir": "data", "dataset_name": "cmd",
            "abundance_file": "x.tsv", "metadata_file": "x.tsv",
        })
        assert isinstance(ds, CuratedMetagenomicDataset)

    def test_name_alias_human_microbiome(self):
        ds = load_dataset("human_microbiome", {
            "data_dir": "data", "dataset_name": "hmp",
            "biom_file": "x.biom", "metadata_file": "x.tsv",
            "min_reads_per_sample": 1000,
        })
        assert isinstance(ds, HumanMicrobiomeDataset)

    def test_fallback_to_config_driven(self):
        """Unknown name + no loader + conditions -> ConfigDrivenDataset."""
        ds = load_dataset("my_cohort", {
            "data_dir": "data", "dataset_name": "my_cohort",
            "abundance_file": "x.tsv", "metadata_file": "x.tsv",
            "conditions": {
                "IBD": {"column": "dx", "case_values": ["ibd"], "control_values": ["ctrl"]},
            },
        })
        assert isinstance(ds, ConfigDrivenDataset)

    def test_unknown_name_no_conditions_raises(self):
        with pytest.raises(ValueError, match="no specialized loader"):
            load_dataset("totally_unknown", {
                "data_dir": "data", "dataset_name": "x",
                "abundance_file": "x.tsv", "metadata_file": "x.tsv",
            })


# ---------------------------------------------------------------------------
# AGP
# ---------------------------------------------------------------------------

class TestAGPDataset:
    @pytest.fixture
    def agp_config(self):
        return {
            "data_dir": "data",
            "dataset_name": "agp",
            "biom_file": "AGP.data.biom",
            "metadata_file": "AGP-metadata.tsv",
            "phylogeny_file": "phylogeny.nwk",
            "min_reads_per_sample": 1000,
            "min_feature_prevalence": 0.001,
        }

    def test_init(self, agp_config):
        ds = AmericanGutDataset(agp_config)
        assert ds.name == "agp"
        assert ds.data_dir == Path("data/agp")

    def test_diseases_list(self, agp_config):
        ds = AmericanGutDataset(agp_config)
        assert "IBD" in ds.list_available_conditions()
        assert len(ds.list_available_conditions()) == 10

    def test_unknown_disease_raises(self, agp_config):
        ds = AmericanGutDataset(agp_config)
        with pytest.raises(ValueError, match="Unknown disease"):
            ds.get_disease_labels("Nonexistent")

    def test_missing_biom_raises(self, agp_config):
        ds = AmericanGutDataset(agp_config)
        with pytest.raises(FileNotFoundError):
            ds.load_abundance_data()

    def test_missing_metadata_raises(self, agp_config):
        ds = AmericanGutDataset(agp_config)
        with pytest.raises(FileNotFoundError):
            ds.load_metadata()

    def test_missing_phylogeny_raises(self, agp_config):
        ds = AmericanGutDataset(agp_config)
        with pytest.raises(FileNotFoundError):
            ds.load_phylogeny()


# ---------------------------------------------------------------------------
# CMD
# ---------------------------------------------------------------------------

class TestCMDDataset:
    @pytest.fixture
    def cmd_config(self):
        return {
            "data_dir": "data",
            "dataset_name": "cmd",
            "abundance_file": "merged_abundance.tsv",
            "metadata_file": "merged_metadata.tsv",
        }

    def test_init(self, cmd_config):
        ds = CuratedMetagenomicDataset(cmd_config)
        assert ds.name == "cmd"
        assert ds.data_dir == Path("data/curatedMetagenomicData")

    def test_diseases_list(self, cmd_config):
        ds = CuratedMetagenomicDataset(cmd_config)
        conditions = ds.list_available_conditions()
        assert "CRC" in conditions
        assert "IBD" in conditions
        assert "T2D" in conditions

    def test_phylogeny_returns_none(self, cmd_config):
        ds = CuratedMetagenomicDataset(cmd_config)
        assert ds.load_phylogeny() is None

    def test_unknown_disease_raises(self, cmd_config):
        ds = CuratedMetagenomicDataset(cmd_config)
        with pytest.raises(ValueError, match="Unknown disease"):
            ds.get_disease_labels("Nonexistent")

    def test_missing_abundance_raises(self):
        cfg = {
            "data_dir": "/tmp/nonexistent_biomeml_test_dir",
            "dataset_name": "cmd",
            "abundance_file": "merged_abundance.tsv",
            "metadata_file": "merged_metadata.tsv",
        }
        ds = CuratedMetagenomicDataset(cfg)
        with pytest.raises(FileNotFoundError):
            ds.load_abundance_data()

    def test_studies_for_disease(self, cmd_config):
        ds = CuratedMetagenomicDataset(cmd_config)
        ibd_studies = ds.get_studies_for_disease("IBD")
        assert "LloydPriceJ_2019" in ibd_studies
        assert len(ibd_studies) == 3


# ---------------------------------------------------------------------------
# HMP
# ---------------------------------------------------------------------------

class TestHMPDataset:
    @pytest.fixture
    def hmp_config(self):
        return {
            "data_dir": "data",
            "dataset_name": "hmp",
            "biom_file": "ibdmdb_abundance.biom",
            "metadata_file": "ibdmdb_metadata.tsv",
            "min_reads_per_sample": 1000,
        }

    def test_init(self, hmp_config):
        ds = HumanMicrobiomeDataset(hmp_config)
        assert ds.name == "hmp"
        assert ds.data_dir == Path("data/hmp")

    def test_conditions(self, hmp_config):
        ds = HumanMicrobiomeDataset(hmp_config)
        assert "IBD" in ds.list_available_conditions()

    def test_unknown_condition_raises(self, hmp_config):
        ds = HumanMicrobiomeDataset(hmp_config)
        with pytest.raises((ValueError, FileNotFoundError)):
            ds.get_disease_labels("Nonexistent")


# ---------------------------------------------------------------------------
# ConfigDrivenDataset
# ---------------------------------------------------------------------------

class TestConfigDrivenDataset:
    """Comprehensive tests for the generic config-driven loader."""

    @pytest.fixture
    def tmpdir(self, tmp_path):
        data_dir = tmp_path / "my_study"
        data_dir.mkdir()
        return tmp_path, data_dir

    def _write_abundance(self, data_dir: Path, fmt='tsv'):
        abundance = pd.DataFrame(
            np.random.rand(10, 20),
            index=[f"S{i}" for i in range(10)],
            columns=[f"OTU_{j}" for j in range(20)],
        )
        path = data_dir / f"abundance.{fmt}"
        sep = '\t' if fmt == 'tsv' else ','
        abundance.to_csv(path, sep=sep)
        return path

    def _write_metadata(self, data_dir: Path, extra_cols=None):
        meta = pd.DataFrame({
            'host_age': np.random.randint(20, 80, 10),
            'host_sex': np.random.choice(['M', 'F'], 10),
            'diagnosis': ['ibd'] * 5 + ['control'] * 5,
        }, index=[f"S{i}" for i in range(10)])
        meta.index.name = 'sample_id'
        if extra_cols:
            for col, vals in extra_cols.items():
                meta[col] = vals
        path = data_dir / "metadata.tsv"
        meta.to_csv(path, sep='\t')
        return path

    def _base_config(self, tmp_path):
        return {
            'dataset_name': 'my_study',
            'data_dir': str(tmp_path),
            'abundance_file': 'abundance.tsv',
            'metadata_file': 'metadata.tsv',
            'columns': {
                'age': 'host_age',
                'sex': 'host_sex',
            },
            'conditions': {
                'IBD': {
                    'column': 'diagnosis',
                    'case_values': ['ibd'],
                    'control_values': ['control', 'healthy'],
                },
            },
        }

    # --- init ---

    def test_init_with_conditions(self, tmpdir):
        tmp_path, _ = tmpdir
        cfg = self._base_config(tmp_path)
        ds = ConfigDrivenDataset(cfg)
        assert ds.name == 'my_study'
        assert ds.list_available_conditions() == ['IBD']

    def test_init_no_conditions_still_works(self, tmpdir):
        tmp_path, _ = tmpdir
        cfg = self._base_config(tmp_path)
        cfg.pop('conditions')
        ds = ConfigDrivenDataset(cfg)
        assert ds.list_available_conditions() == []

    def test_init_no_columns_still_works(self, tmpdir):
        tmp_path, _ = tmpdir
        cfg = self._base_config(tmp_path)
        cfg.pop('columns')
        ds = ConfigDrivenDataset(cfg)
        assert ds.column_map == {}

    def test_missing_abundance_file_raises(self, tmpdir):
        tmp_path, _ = tmpdir
        cfg = self._base_config(tmp_path)
        ds = ConfigDrivenDataset(cfg)
        with pytest.raises(FileNotFoundError, match="Abundance file not found"):
            ds.load_abundance_data()

    def test_missing_metadata_file_raises(self, tmpdir):
        tmp_path, data_dir = tmpdir
        self._write_abundance(data_dir)
        cfg = self._base_config(tmp_path)
        ds = ConfigDrivenDataset(cfg)
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            ds.load_metadata()

    # --- load abundance ---

    def test_load_abundance_tsv(self, tmpdir):
        tmp_path, data_dir = tmpdir
        self._write_abundance(data_dir, fmt='tsv')
        cfg = self._base_config(tmp_path)
        ds = ConfigDrivenDataset(cfg)
        df = ds.load_abundance_data()
        assert df.shape == (10, 20)

    def test_load_abundance_csv(self, tmpdir):
        tmp_path, data_dir = tmpdir
        self._write_abundance(data_dir, fmt='csv')
        cfg = self._base_config(tmp_path)
        cfg['abundance_file'] = 'abundance.csv'
        ds = ConfigDrivenDataset(cfg)
        df = ds.load_abundance_data()
        assert df.shape == (10, 20)

    def test_abundance_caching(self, tmpdir):
        tmp_path, data_dir = tmpdir
        self._write_abundance(data_dir)
        cfg = self._base_config(tmp_path)
        ds = ConfigDrivenDataset(cfg)
        df1 = ds.load_abundance_data()
        df2 = ds.load_abundance_data()
        assert df1 is df2

    # --- load metadata + column mapping ---

    def test_load_metadata_renames_columns(self, tmpdir):
        tmp_path, data_dir = tmpdir
        self._write_metadata(data_dir)
        cfg = self._base_config(tmp_path)
        ds = ConfigDrivenDataset(cfg)
        meta = ds.load_metadata()
        assert 'age' in meta.columns
        assert 'sex' in meta.columns

    def test_load_metadata_preserves_unmapped_columns(self, tmpdir):
        tmp_path, data_dir = tmpdir
        self._write_metadata(data_dir)
        cfg = self._base_config(tmp_path)
        ds = ConfigDrivenDataset(cfg)
        meta = ds.load_metadata()
        assert 'diagnosis' in meta.columns

    def test_load_metadata_sample_id_column(self, tmpdir):
        tmp_path, data_dir = tmpdir
        meta = pd.DataFrame({
            'SampleID': [f"S{i}" for i in range(5)],
            'dx': ['ibd'] * 3 + ['ctrl'] * 2,
        })
        meta.to_csv(data_dir / "metadata.tsv", sep='\t', index=False)
        cfg = self._base_config(tmp_path)
        cfg['columns'] = {'sample_id': 'SampleID'}
        cfg['conditions'] = {
            'IBD': {'column': 'dx', 'case_values': ['ibd'], 'control_values': ['ctrl']},
        }
        ds = ConfigDrivenDataset(cfg)
        m = ds.load_metadata()
        assert m.index.name == 'SampleID'

    # --- get_disease_labels ---

    def test_get_disease_labels_basic(self, tmpdir):
        tmp_path, data_dir = tmpdir
        self._write_abundance(data_dir)
        self._write_metadata(data_dir)
        cfg = self._base_config(tmp_path)
        ds = ConfigDrivenDataset(cfg)
        ids, labels = ds.get_disease_labels('IBD')
        assert len(ids) == 10
        assert labels.sum() == 5
        assert (labels == 0).sum() == 5

    def test_get_disease_labels_unknown_condition_raises(self, tmpdir):
        tmp_path, _ = tmpdir
        cfg = self._base_config(tmp_path)
        ds = ConfigDrivenDataset(cfg)
        with pytest.raises(ValueError, match="Unknown condition"):
            ds.get_disease_labels('Nonexistent')

    def test_get_disease_labels_missing_column_raises(self, tmpdir):
        tmp_path, data_dir = tmpdir
        self._write_metadata(data_dir)
        cfg = self._base_config(tmp_path)
        cfg['conditions']['Bad'] = {
            'column': 'nonexistent_col',
            'case_values': ['x'], 'control_values': ['y'],
        }
        ds = ConfigDrivenDataset(cfg)
        with pytest.raises(ValueError, match="not found in metadata"):
            ds.get_disease_labels('Bad')

    def test_get_disease_labels_no_matches_raises(self, tmpdir):
        tmp_path, data_dir = tmpdir
        self._write_metadata(data_dir)
        cfg = self._base_config(tmp_path)
        cfg['conditions']['Empty'] = {
            'column': 'diagnosis',
            'case_values': ['nonexistent_value'],
            'control_values': ['also_nonexistent'],
        }
        ds = ConfigDrivenDataset(cfg)
        with pytest.raises(ValueError, match="No valid case/control"):
            ds.get_disease_labels('Empty')

    def test_get_disease_labels_case_insensitive(self, tmpdir):
        tmp_path, data_dir = tmpdir
        meta = pd.DataFrame({
            'dx': ['IBD', 'Ibd', 'CONTROL', 'Control'],
        }, index=[f"S{i}" for i in range(4)])
        meta.index.name = 'sid'
        (data_dir / "metadata.tsv").write_text(meta.to_csv(sep='\t'))
        cfg = self._base_config(tmp_path)
        cfg['conditions'] = {
            'IBD': {'column': 'dx', 'case_values': ['ibd'], 'control_values': ['control']},
        }
        ds = ConfigDrivenDataset(cfg)
        ids, labels = ds.get_disease_labels('IBD')
        assert len(ids) == 4
        assert labels.sum() == 2

    # --- multiple conditions ---

    def test_multiple_conditions(self, tmpdir):
        tmp_path, data_dir = tmpdir
        extra = {'diabetes': ['t2d'] * 3 + ['healthy'] * 7}
        self._write_abundance(data_dir)
        self._write_metadata(data_dir, extra_cols=extra)
        cfg = self._base_config(tmp_path)
        cfg['conditions']['Diabetes'] = {
            'column': 'diabetes',
            'case_values': ['t2d'],
            'control_values': ['healthy'],
        }
        ds = ConfigDrivenDataset(cfg)
        assert set(ds.list_available_conditions()) == {'IBD', 'Diabetes'}
        ids, labels = ds.get_disease_labels('Diabetes')
        assert labels.sum() == 3

    # --- get_abundance_for_samples ---

    def test_get_abundance_for_samples(self, tmpdir):
        tmp_path, data_dir = tmpdir
        self._write_abundance(data_dir)
        self._write_metadata(data_dir)
        cfg = self._base_config(tmp_path)
        ds = ConfigDrivenDataset(cfg)
        df = ds.get_abundance_for_samples(['S0', 'S1', 'S2'])
        assert df.shape[0] == 3
        row_sums = df.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_get_abundance_no_overlap_raises(self, tmpdir):
        tmp_path, data_dir = tmpdir
        self._write_abundance(data_dir)
        cfg = self._base_config(tmp_path)
        ds = ConfigDrivenDataset(cfg)
        with pytest.raises(ValueError, match="No requested samples"):
            ds.get_abundance_for_samples(['NONEXISTENT_1', 'NONEXISTENT_2'])

    # --- phylogeny ---

    def test_phylogeny_none_when_not_configured(self, tmpdir):
        tmp_path, _ = tmpdir
        cfg = self._base_config(tmp_path)
        ds = ConfigDrivenDataset(cfg)
        assert ds.load_phylogeny() is None

    def test_phylogeny_none_when_file_missing(self, tmpdir):
        tmp_path, _ = tmpdir
        cfg = self._base_config(tmp_path)
        cfg['phylogeny_file'] = 'nonexistent.nwk'
        ds = ConfigDrivenDataset(cfg)
        assert ds.load_phylogeny() is None

    # --- validate ---

    def test_validate_success(self, tmpdir):
        tmp_path, data_dir = tmpdir
        self._write_abundance(data_dir)
        self._write_metadata(data_dir)
        cfg = self._base_config(tmp_path)
        ds = ConfigDrivenDataset(cfg)
        assert ds.validate() is True

    def test_validate_no_overlap_raises(self, tmpdir):
        tmp_path, data_dir = tmpdir
        abundance = pd.DataFrame(
            np.random.rand(3, 5),
            index=['X0', 'X1', 'X2'],
            columns=[f"OTU_{j}" for j in range(5)],
        )
        abundance.to_csv(data_dir / "abundance.tsv", sep='\t')
        self._write_metadata(data_dir)
        cfg = self._base_config(tmp_path)
        ds = ConfigDrivenDataset(cfg)
        with pytest.raises(ValueError, match="No sample ID overlap"):
            ds.validate()

    # --- info ---

    def test_info(self, tmpdir):
        tmp_path, _ = tmpdir
        cfg = self._base_config(tmp_path)
        ds = ConfigDrivenDataset(cfg)
        info = ds.info()
        assert info['name'] == 'my_study'
        assert 'column_mappings' in info

    # --- helpers ---

    # --- column mapping + condition E2E ---

    def test_condition_uses_renamed_column(self, tmpdir):
        """End-to-end: metadata has 'dx', columns maps disease_status->dx, condition uses mapped name."""
        tmp_path, data_dir = tmpdir
        self._write_abundance(data_dir)
        meta = pd.DataFrame({
            'raw_diag': ['ibd'] * 4 + ['healthy'] * 6,
        }, index=[f"S{i}" for i in range(10)])
        meta.index.name = 'sample_id'
        meta.to_csv(data_dir / "metadata.tsv", sep='\t')

        cfg = self._base_config(tmp_path)
        cfg['columns'] = {'disease_status': 'raw_diag'}
        cfg['conditions'] = {
            'IBD': {
                'column': 'disease_status',
                'case_values': ['ibd'],
                'control_values': ['healthy'],
            },
        }
        ds = ConfigDrivenDataset(cfg)
        ids, labels = ds.get_disease_labels('IBD')
        assert len(ids) == 10
        assert labels.sum() == 4

    def test_condition_uses_original_column_name(self, tmpdir):
        """Condition can still refer to original column name if not in column_map."""
        tmp_path, data_dir = tmpdir
        self._write_abundance(data_dir)
        self._write_metadata(data_dir)
        cfg = self._base_config(tmp_path)
        cfg['conditions'] = {
            'IBD': {
                'column': 'diagnosis',
                'case_values': ['ibd'],
                'control_values': ['control'],
            },
        }
        ds = ConfigDrivenDataset(cfg)
        ids, labels = ds.get_disease_labels('IBD')
        assert len(ids) == 10

    def test_detect_separator_tsv(self, tmp_path):
        p = tmp_path / "test.tsv"
        p.write_text("a\tb\tc\n1\t2\t3\n")
        assert _detect_separator(p) == '\t'

    def test_detect_separator_csv(self, tmp_path):
        p = tmp_path / "test.csv"
        p.write_text("a,b,c\n1,2,3\n")
        assert _detect_separator(p) == ','


# ---------------------------------------------------------------------------
# Custom (legacy, back-compat)
# ---------------------------------------------------------------------------

class TestCustomDataset:
    @pytest.fixture
    def custom_config(self):
        return {
            "data_dir": "data",
            "dataset_name": "my_test",
            "abundance_file": "abundance.tsv",
            "metadata_file": "metadata.tsv",
            "target_column": "disease",
            "case_values": ["1"],
            "control_values": ["0"],
        }

    def test_init(self, custom_config):
        ds = CustomDataset(custom_config)
        assert ds.name == "my_test"
        assert ds.target_col == "disease"

    def test_missing_abundance_raises(self, custom_config):
        ds = CustomDataset(custom_config)
        with pytest.raises(FileNotFoundError):
            ds.load_abundance_data()

    def test_missing_metadata_raises(self, custom_config):
        ds = CustomDataset(custom_config)
        with pytest.raises(FileNotFoundError):
            ds.load_metadata()

    def test_phylogeny_returns_none_if_missing(self, custom_config):
        ds = CustomDataset(custom_config)
        assert ds.load_phylogeny() is None


# ---------------------------------------------------------------------------
# load_dataset factory
# ---------------------------------------------------------------------------

class TestLoadDatasetFactory:
    def test_load_agp(self):
        ds = load_dataset("agp", {
            "data_dir": "data", "dataset_name": "agp",
            "biom_file": "x.biom", "metadata_file": "x.tsv",
            "phylogeny_file": "x.nwk", "min_reads_per_sample": 1000,
            "min_feature_prevalence": 0.001,
        })
        assert isinstance(ds, AmericanGutDataset)

    def test_load_cmd(self):
        ds = load_dataset("cmd", {
            "data_dir": "data", "dataset_name": "cmd",
            "abundance_file": "x.tsv", "metadata_file": "x.tsv",
        })
        assert isinstance(ds, CuratedMetagenomicDataset)

    def test_case_insensitive(self):
        ds = load_dataset("CMD", {
            "data_dir": "data", "dataset_name": "cmd",
            "abundance_file": "x.tsv", "metadata_file": "x.tsv",
        })
        assert isinstance(ds, CuratedMetagenomicDataset)

    def test_unknown_with_conditions_goes_config_driven(self):
        ds = load_dataset("brand_new_study", {
            "data_dir": "data", "dataset_name": "brand_new_study",
            "abundance_file": "x.tsv", "metadata_file": "x.tsv",
            "conditions": {
                "Asthma": {
                    "column": "asthma_status",
                    "case_values": ["yes"],
                    "control_values": ["no"],
                },
            },
        })
        assert isinstance(ds, ConfigDrivenDataset)
        assert ds.list_available_conditions() == ["Asthma"]


# ---------------------------------------------------------------------------
# datasets_config/ YAML validation
# ---------------------------------------------------------------------------

class TestDatasetConfigYAMLFiles:
    """Validate that every YAML file in datasets_config/ is loadable and consistent."""

    @pytest.fixture
    def configs_dir(self):
        return Path(__file__).resolve().parent.parent / "datasets_config"

    def _load_yaml(self, path):
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)

    def test_agp_yaml_has_required_keys(self, configs_dir):
        cfg = self._load_yaml(configs_dir / "agp.yaml")
        assert cfg['dataset_name'] == 'agp'
        assert 'data_dir' in cfg
        assert 'biom_file' in cfg or 'abundance_file' in cfg
        assert 'metadata_file' in cfg
        assert 'conditions' in cfg
        assert 'IBD' in cfg['conditions']
        assert 'columns' in cfg

    def test_cmd_yaml_has_required_keys(self, configs_dir):
        cfg = self._load_yaml(configs_dir / "cmd.yaml")
        assert cfg['dataset_name'] == 'cmd'
        assert 'abundance_file' in cfg
        assert 'metadata_file' in cfg
        assert 'conditions' in cfg
        assert 'CRC' in cfg['conditions']
        assert 'IBD' in cfg['conditions']
        assert 'columns' in cfg

    def test_hmp_yaml_has_required_keys(self, configs_dir):
        cfg = self._load_yaml(configs_dir / "hmp.yaml")
        assert cfg['dataset_name'] == 'hmp'
        assert 'metadata_file' in cfg
        assert 'conditions' in cfg
        assert 'IBD' in cfg['conditions']
        assert 'columns' in cfg

    def test_all_conditions_have_required_fields(self, configs_dir):
        import yaml
        for yaml_file in configs_dir.glob("*.yaml"):
            if 'template' in yaml_file.name:
                continue
            cfg = self._load_yaml(yaml_file)
            conditions = cfg.get('conditions', {})
            for cond_name, cond_def in conditions.items():
                assert 'column' in cond_def, (
                    f"{yaml_file.name}: condition '{cond_name}' missing 'column'"
                )
                assert 'case_values' in cond_def, (
                    f"{yaml_file.name}: condition '{cond_name}' missing 'case_values'"
                )
                assert 'control_values' in cond_def, (
                    f"{yaml_file.name}: condition '{cond_name}' missing 'control_values'"
                )
                assert len(cond_def['case_values']) > 0
                assert len(cond_def['control_values']) > 0

    def test_agp_yaml_instantiates_via_loader_key(self, configs_dir):
        cfg = self._load_yaml(configs_dir / "agp.yaml")
        ds = load_dataset(cfg['dataset_name'], cfg)
        assert isinstance(ds, AmericanGutDataset)

    def test_cmd_yaml_instantiates_via_loader_key(self, configs_dir):
        cfg = self._load_yaml(configs_dir / "cmd.yaml")
        ds = load_dataset(cfg['dataset_name'], cfg)
        assert isinstance(ds, CuratedMetagenomicDataset)

    def test_hmp_yaml_instantiates_via_loader_key(self, configs_dir):
        cfg = self._load_yaml(configs_dir / "hmp.yaml")
        ds = load_dataset(cfg['dataset_name'], cfg)
        assert isinstance(ds, HumanMicrobiomeDataset)

    def test_agp_yaml_without_loader_goes_config_driven(self, configs_dir):
        """If loader: key is removed, AGP YAML should work as config-driven."""
        cfg = self._load_yaml(configs_dir / "agp.yaml")
        cfg.pop('loader', None)
        cfg['dataset_name'] = 'my_agp_study'
        cfg['abundance_file'] = cfg.pop('biom_file', 'abundance.tsv')
        ds = load_dataset('my_agp_study', cfg)
        assert isinstance(ds, ConfigDrivenDataset)
        assert 'IBD' in ds.list_available_conditions()

    def test_custom_template_is_parseable(self, configs_dir):
        cfg = self._load_yaml(configs_dir / "custom_template.yaml")
        assert 'dataset_name' in cfg
        assert 'conditions' in cfg
