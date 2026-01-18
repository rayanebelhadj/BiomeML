"""Tests for src/edge_weights.py — all strategies, edge cases, known bugs."""

import math
import pytest
import numpy as np

from src.edge_weights import (
    identity, inverse, exponential, binary,
    abundance_product, abundance_geometric, abundance_log,
    abundance_min, abundance_max,
    get_edge_weight_function,
    compute_edge_weights_for_graph,
    list_strategies,
    EDGE_WEIGHT_STRATEGIES,
)


# =========================================================================
# Individual strategy correctness
# =========================================================================

class TestStrategies:
    def test_identity(self):
        assert identity(0.5) == 0.5
        assert identity(0.0) == 0.0

    def test_inverse(self):
        assert inverse(0.5) == pytest.approx(1.0 / (0.5 + 1e-8))
        assert inverse(0.0) == pytest.approx(1.0 / 1e-8)

    def test_exponential(self):
        assert exponential(0.0) == pytest.approx(1.0)
        assert exponential(1.0) == pytest.approx(math.exp(-1))

    def test_binary(self):
        assert binary(0.5) == 1.0
        assert binary(999.0) == 1.0

    def test_abundance_product(self):
        w = abundance_product(0.5, 0.1, 0.2)
        expected = (1.0 / (0.5 + 1e-8)) * 0.1 * 0.2
        assert w == pytest.approx(expected)

    def test_abundance_product_none_fallback(self):
        w = abundance_product(0.5, None, None)
        assert w == pytest.approx(1.0 / (0.5 + 1e-8))

    def test_abundance_geometric(self):
        w = abundance_geometric(0.5, 0.1, 0.2)
        expected = (1.0 / (0.5 + 1e-8)) * np.sqrt(0.1 * 0.2 + 1e-8)
        assert w == pytest.approx(expected)

    def test_abundance_log(self):
        w = abundance_log(0.5, 0.1, 0.2)
        expected = (1.0 / (0.5 + 1e-8)) * np.log1p(0.1) * np.log1p(0.2)
        assert w == pytest.approx(expected)

    def test_abundance_min(self):
        w = abundance_min(0.5, 0.1, 0.2)
        expected = (1.0 / (0.5 + 1e-8)) * 0.1
        assert w == pytest.approx(expected)

    def test_abundance_max(self):
        w = abundance_max(0.5, 0.1, 0.2)
        expected = (1.0 / (0.5 + 1e-8)) * 0.2
        assert w == pytest.approx(expected)


# =========================================================================
# Strategy properties — must hold for ALL strategies
# =========================================================================

class TestStrategyProperties:
    @pytest.mark.parametrize("name", list(EDGE_WEIGHT_STRATEGIES.keys()))
    def test_positive_distance_produces_finite(self, name):
        fn = EDGE_WEIGHT_STRATEGIES[name]
        if name.startswith('abundance_'):
            w = fn(0.5, 0.1, 0.2)
        else:
            w = fn(0.5)
        assert np.isfinite(w), f"{name}(0.5) is not finite: {w}"

    @pytest.mark.parametrize("name", list(EDGE_WEIGHT_STRATEGIES.keys()))
    def test_non_negative_output(self, name):
        fn = EDGE_WEIGHT_STRATEGIES[name]
        if name.startswith('abundance_'):
            w = fn(0.5, 0.1, 0.2)
        else:
            w = fn(0.5)
        assert w >= 0, f"{name}(0.5) is negative: {w}"

    @pytest.mark.parametrize("name", list(EDGE_WEIGHT_STRATEGIES.keys()))
    def test_zero_distance_produces_finite(self, name):
        fn = EDGE_WEIGHT_STRATEGIES[name]
        if name.startswith('abundance_'):
            w = fn(0.0, 0.1, 0.2)
        else:
            w = fn(0.0)
        assert np.isfinite(w), f"{name}(0.0) is not finite: {w}"

    @pytest.mark.parametrize("name", list(EDGE_WEIGHT_STRATEGIES.keys()))
    def test_large_distance_produces_finite(self, name):
        fn = EDGE_WEIGHT_STRATEGIES[name]
        if name.startswith('abundance_'):
            w = fn(1000.0, 0.1, 0.2)
        else:
            w = fn(1000.0)
        assert np.isfinite(w), f"{name}(1000) is not finite: {w}"

    def test_inverse_monotonically_decreasing(self):
        """Closer distance → higher weight."""
        assert inverse(0.1) > inverse(0.5) > inverse(1.0) > inverse(10.0)

    def test_exponential_monotonically_decreasing(self):
        assert exponential(0.1) > exponential(0.5) > exponential(1.0) > exponential(10.0)

    def test_abundance_zero_gives_zero(self):
        """If either abundance is 0, product/min should be 0 (or near-zero)."""
        assert abundance_product(0.5, 0.0, 0.5) == pytest.approx(0.0)
        assert abundance_min(0.5, 0.0, 0.5) == pytest.approx(0.0)

    def test_abundance_symmetry(self):
        """abundance strategies should be symmetric in a1, a2."""
        assert abundance_product(0.5, 0.1, 0.2) == pytest.approx(abundance_product(0.5, 0.2, 0.1))
        assert abundance_geometric(0.5, 0.1, 0.2) == pytest.approx(abundance_geometric(0.5, 0.2, 0.1))
        assert abundance_log(0.5, 0.1, 0.2) == pytest.approx(abundance_log(0.5, 0.2, 0.1))


# =========================================================================
# get_edge_weight_function
# =========================================================================

class TestFactory:
    def test_all_strategies_loadable(self):
        for name in list_strategies():
            fn = get_edge_weight_function(name)
            assert callable(fn)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            get_edge_weight_function("nonexistent")

    def test_list_strategies_count(self):
        assert len(list_strategies()) == len(EDGE_WEIGHT_STRATEGIES)
        assert len(list_strategies()) == 9


# =========================================================================
# compute_edge_weights_for_graph — comprehensive
# =========================================================================

class TestComputeEdgeWeights:
    def test_basic(self):
        edges = [("A", "B"), ("B", "C")]
        distances = {("A", "B"): 0.5, ("B", "C"): 0.3}
        abundances = {"A": 0.1, "B": 0.2, "C": 0.3}
        result = compute_edge_weights_for_graph(edges, distances, abundances, strategy='inverse')
        assert ("A", "B") in result
        assert ("B", "C") in result

    def test_reverse_lookup(self):
        edges = [("A", "B")]
        distances = {("B", "A"): 0.5}
        abundances = {}
        result = compute_edge_weights_for_graph(edges, distances, abundances, strategy='inverse')
        assert result[("A", "B")] == pytest.approx(1.0 / (0.5 + 1e-8))

    def test_missing_distance_defaults_to_1(self):
        edges = [("A", "B")]
        distances = {}
        abundances = {}
        result = compute_edge_weights_for_graph(edges, distances, abundances, strategy='identity')
        assert result[("A", "B")] == pytest.approx(1.0)

    @pytest.mark.parametrize("strategy", [
        'inverse', 'exponential', 'binary', 'identity',
        'abundance_product', 'abundance_geometric', 'abundance_log',
        'abundance_min', 'abundance_max',
    ])
    def test_all_strategies_work_with_compute(self, strategy):
        edges = [("A", "B")]
        distances = {("A", "B"): 0.5}
        abundances = {"A": 0.1, "B": 0.2}
        result = compute_edge_weights_for_graph(edges, distances, abundances, strategy=strategy)
        assert ("A", "B") in result
        assert np.isfinite(result[("A", "B")])

    def test_zero_distance_preserved(self):
        """d=0.0 must be returned as 0.0, not treated as missing (was a bug with `or`)."""
        edges = [("A", "B")]
        distances = {("A", "B"): 0.0}
        abundances = {}
        result = compute_edge_weights_for_graph(edges, distances, abundances, strategy='identity')
        assert result[("A", "B")] == pytest.approx(0.0), \
            "d=0.0 must be preserved, not treated as missing"

    def test_zero_distance_forward_only(self):
        """d=0.0 in forward direction only must be used (not fall to reverse/default)."""
        edges = [("A", "B")]
        distances = {("A", "B"): 0.0}
        abundances = {}
        result = compute_edge_weights_for_graph(edges, distances, abundances, strategy='identity')
        assert result[("A", "B")] == pytest.approx(0.0)
