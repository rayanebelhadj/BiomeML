"""Tests for src/feature_loader.py — clinical feature loading and normalization."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.feature_loader import load_features


def _make_config(disease='IBD', enable=False, features=None, use_clinical_features=False):
    """Build a valid config dict for load_features with all required keys."""
    if features is None:
        features = {'use_age': True, 'use_sex': True, 'use_bmi': True, 'use_antibiotics': False}
    return {
        'data_extraction': {
            'disease_criteria': {'disease': disease},
            'clinical_features': {
                'enable': enable,
                'features': features,
            },
        },
        'model_training': {
            'architecture': {'use_clinical_features': use_clinical_features},
        },
    }


class TestLoadFeaturesDisabled:
    """When clinical features are disabled, load_features should return None."""

    def test_none_config_raises(self):
        with pytest.raises(ValueError, match="must not be None"):
            load_features(None, "/tmp")

    def test_empty_config_raises(self):
        with pytest.raises(KeyError):
            load_features({}, "/tmp")

    def test_clinical_disabled_explicitly(self):
        config = _make_config(enable=False, use_clinical_features=False)
        result = load_features(config, "/tmp")
        assert result['use_clinical'] is False


class TestLoadFeaturesMissingFile:
    """When clinical is enabled but file missing, should raise."""

    def test_missing_metadata_raises(self, tmp_path):
        config = _make_config(enable=True)
        with pytest.raises(FileNotFoundError, match="metadata file not found"):
            load_features(config, str(tmp_path))


class TestLoadFeaturesNormalization:
    """Verify normalization produces values in [0, 1]."""

    def test_age_normalization(self, tmp_path):
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        df = pd.DataFrame({
            'age_cat': ['20s', '30s', '40s', '50s', '60s'],
        }, index=[f'sample_{i}' for i in range(5)])
        df.to_csv(meta_dir / "AGP_IBD_metadata.txt", sep='\t')

        config = _make_config(
            enable=True,
            features={'use_age': True, 'use_sex': False, 'use_bmi': False, 'use_antibiotics': False},
        )
        result = load_features(config, str(tmp_path))
        assert result['use_clinical'] is True
        assert result['clinical_dim'] == 1
        assert result['clinical'] is not None

        vals = result['clinical'].values.flatten()
        assert np.all(vals >= 0.0)
        assert np.all(vals <= 1.0)
        assert not np.any(np.isnan(vals))

    def test_sex_normalization(self, tmp_path):
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        df = pd.DataFrame({
            'sex': ['male', 'female', 'Male', 'Female', 'male'],
        }, index=[f's{i}' for i in range(5)])
        df.to_csv(meta_dir / "AGP_IBD_metadata.txt", sep='\t')

        config = _make_config(
            enable=True,
            features={'use_age': False, 'use_sex': True, 'use_bmi': False, 'use_antibiotics': False},
        )
        result = load_features(config, str(tmp_path))
        assert result['use_clinical'] is True
        vals = result['clinical'].values.flatten()
        assert np.all(np.isfinite(vals))

    def test_bmi_normalization(self, tmp_path):
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        df = pd.DataFrame({
            'bmi_cat': ['Underweight', 'Normal', 'Overweight', 'Obese', 'Normal'],
        }, index=[f's{i}' for i in range(5)])
        df.to_csv(meta_dir / "AGP_IBD_metadata.txt", sep='\t')

        config = _make_config(
            enable=True,
            features={'use_age': False, 'use_sex': False, 'use_bmi': True, 'use_antibiotics': False},
        )
        result = load_features(config, str(tmp_path))
        assert result['use_clinical'] is True
        vals = result['clinical'].values.flatten()
        assert np.all(vals >= 0.0)
        assert np.all(vals <= 1.0)


class TestLoadFeaturesNaN:
    """NaN in metadata should be handled (filled), not propagated."""

    def test_nan_age_filled(self, tmp_path):
        meta_dir = tmp_path / "metadata"
        meta_dir.mkdir()
        df = pd.DataFrame({
            'age_cat': ['20s', np.nan, '40s', 'unknown_value', '60s'],
        }, index=[f's{i}' for i in range(5)])
        df.to_csv(meta_dir / "AGP_IBD_metadata.txt", sep='\t')

        config = _make_config(
            enable=True,
            features={'use_age': True, 'use_sex': False, 'use_bmi': False, 'use_antibiotics': False},
        )
        result = load_features(config, str(tmp_path))
        vals = result['clinical'].values.flatten()
        assert not np.any(np.isnan(vals)), "NaN values leaked through to clinical features"
