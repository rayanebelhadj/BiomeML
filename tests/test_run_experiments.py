"""Tests for scripts/run_experiments.py dataset propagation and extraction."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_experiments import _extract_disease, _extract_dataset_name


class TestExtractDisease:
    def test_from_experiment_override(self):
        assert _extract_disease({}, {"disease": "ibd"}) == "IBD"

    def test_from_base_config(self):
        cfg = {"data_extraction": {"disease_criteria": {"disease": "cancer"}}}
        assert _extract_disease(cfg, {}) == "CANCER"

    def test_override_takes_priority(self):
        cfg = {"data_extraction": {"disease_criteria": {"disease": "cancer"}}}
        assert _extract_disease(cfg, {"disease": "ibd"}) == "IBD"

    def test_missing_everywhere_raises(self):
        with pytest.raises(ValueError, match="Cannot determine disease"):
            _extract_disease({}, {})

    def test_empty_config_raises(self):
        with pytest.raises(ValueError, match="Cannot determine disease"):
            _extract_disease({"data_extraction": {}}, {})


class TestExtractDatasetName:
    def test_from_experiment_override(self):
        assert _extract_dataset_name({}, {"dataset": "cmd"}) == "cmd"

    def test_from_base_config(self):
        cfg = {"dataset": {"name": "hmp"}}
        assert _extract_dataset_name(cfg, {}) == "hmp"

    def test_override_takes_priority(self):
        cfg = {"dataset": {"name": "agp"}}
        assert _extract_dataset_name(cfg, {"dataset": "cmd"}) == "cmd"

    def test_missing_everywhere_raises(self):
        with pytest.raises(ValueError, match="Cannot determine dataset"):
            _extract_dataset_name({}, {})

    def test_empty_dataset_section_raises(self):
        with pytest.raises(ValueError, match="Cannot determine dataset"):
            _extract_dataset_name({"dataset": {}}, {})

    def test_no_dataset_section_raises(self):
        with pytest.raises(ValueError, match="Cannot determine dataset"):
            _extract_dataset_name({"other_key": "value"}, {})

    def test_no_silent_default_to_agp(self):
        """Regression: previously defaulted to 'agp' silently."""
        with pytest.raises(ValueError):
            _extract_dataset_name({"data_extraction": {"disease_criteria": {"disease": "IBD"}}}, {})
