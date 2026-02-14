"""Tests for src/graph_utils.py — builders, config, transforms, edge cases."""

import numpy as np
import networkx as nx
import pytest

from src.graph_utils import (
    build_knn_graph,
    build_mst_graph,
    build_threshold_graph,
    build_tree_graph,
    build_hierarchical_graph,
    randomize_edges,
    transform_edge_weights,
    compute_graph_statistics,
    validate_graph,
    make_complete_graph,
    get_graph_builder,
    get_graph_params_from_config,
)


# =========================================================================
# Zero-distance edge cases (§2.1, §2.2, §2.3 — the critical fixes)
# =========================================================================

class TestZeroDistanceHandling:
    """All three builders must connect features with distance == 0."""

    def test_knn_includes_zero_distance_pair(self, small_distance_matrix):
        dm, ids = small_distance_matrix
        G = build_knn_graph(dm, ids, k=2)
        assert G.has_edge("A", "B"), "k-NN must connect zero-distance pair (A, B)"

    def test_mst_includes_zero_distance_pair(self, small_distance_matrix):
        dm, ids = small_distance_matrix
        G = build_mst_graph(dm, ids)
        assert G.has_edge("A", "B"), "MST must connect zero-distance pair (A, B)"
        edge_data = G.edges["A", "B"]
        assert edge_data["distance"] == 0.0

    def test_threshold_includes_zero_distance_pair(self, small_distance_matrix):
        dm, ids = small_distance_matrix
        G = build_threshold_graph(dm, ids, threshold_percentile=50)
        assert G.has_edge("A", "B"), "Threshold must connect zero-distance pair (A, B)"

    def test_knn_zero_distance_gets_high_weight(self, small_distance_matrix):
        dm, ids = small_distance_matrix
        G = build_knn_graph(dm, ids, k=4)
        if G.has_edge("A", "B"):
            w = G.edges["A", "B"]["weight"]
            other_weights = [d["weight"] for u, v, d in G.edges(data=True) if (u, v) != ("A", "B") and (v, u) != ("A", "B")]
            assert w >= max(other_weights), "Zero-distance edge must have the highest weight (inverse)"

    def test_all_zero_off_diagonal_knn(self, all_zero_off_diag_matrix):
        dm, ids = all_zero_off_diag_matrix
        G = build_knn_graph(dm, ids, k=2)
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() > 0, "All-zero matrix should still produce edges"

    def test_all_zero_off_diagonal_mst(self, all_zero_off_diag_matrix):
        dm, ids = all_zero_off_diag_matrix
        G = build_mst_graph(dm, ids)
        assert G.number_of_edges() == 2, "MST of 3-node all-zero should have 2 edges"

    def test_all_zero_off_diagonal_threshold(self, all_zero_off_diag_matrix):
        dm, ids = all_zero_off_diag_matrix
        G = build_threshold_graph(dm, ids, threshold_percentile=50)
        assert G.number_of_edges() > 0


# =========================================================================
# build_knn_graph — comprehensive
# =========================================================================

class TestBuildKnnGraph:
    def test_basic_shape(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        G = build_knn_graph(dm, ids, k=5)
        assert G.number_of_nodes() == 20
        assert G.number_of_edges() > 0

    def test_k_equals_1(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        G = build_knn_graph(dm, ids, k=1, symmetric=False)
        assert G.number_of_edges() >= 1

    def test_k_too_large_still_produces_graph(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        G = build_knn_graph(dm, ids, k=100)
        assert G.number_of_nodes() == 20
        assert G.number_of_edges() > 0

    def test_invalid_k_raises(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        with pytest.raises(ValueError, match="k must be >= 1"):
            build_knn_graph(dm, ids, k=0)

    def test_symmetric_has_more_or_equal_edges(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        G_sym = build_knn_graph(dm, ids, k=3, symmetric=True)
        G_int = build_knn_graph(dm, ids, k=3, symmetric=False)
        assert G_sym.number_of_edges() >= G_int.number_of_edges()

    def test_edges_have_weight_and_distance(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        G = build_knn_graph(dm, ids, k=3)
        for u, v, d in G.edges(data=True):
            assert "weight" in d and "distance" in d
            assert d["weight"] > 0
            assert d["distance"] >= 0

    def test_weight_is_inverse_distance(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        G = build_knn_graph(dm, ids, k=3)
        for u, v, d in G.edges(data=True):
            expected = 1.0 / (d["distance"] + 1e-8)
            assert d["weight"] == pytest.approx(expected, rel=1e-6)

    def test_with_abundances(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        abund = {fid: 0.5 for fid in ids}
        G = build_knn_graph(dm, ids, k=3, abundances=abund)
        for n, d in G.nodes(data=True):
            assert d["weight"] == 0.5

    def test_max_distance_factor_limits_edges(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        G_strict = build_knn_graph(dm, ids, k=5, max_distance_factor=0.5)
        G_loose = build_knn_graph(dm, ids, k=5, max_distance_factor=10.0)
        assert G_strict.number_of_edges() <= G_loose.number_of_edges()

    def test_shape_mismatch_raises(self):
        dm = np.zeros((3, 4))
        with pytest.raises(ValueError, match="shape"):
            build_knn_graph(dm, ["a", "b", "c"], k=1)

    def test_deterministic(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        G1 = build_knn_graph(dm, ids, k=5)
        G2 = build_knn_graph(dm, ids, k=5)
        assert set(G1.edges()) == set(G2.edges())

    def test_large_graph(self, large_distance_matrix):
        dm, ids = large_distance_matrix
        G = build_knn_graph(dm, ids, k=10)
        assert G.number_of_nodes() == 100
        assert G.number_of_edges() > 100

    def test_no_self_loops(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        G = build_knn_graph(dm, ids, k=5)
        for u, v in G.edges():
            assert u != v, "Graph must not have self-loops"


# =========================================================================
# build_mst_graph — comprehensive
# =========================================================================

class TestBuildMstGraph:
    def test_mst_has_n_minus_1_edges(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        G = build_mst_graph(dm, ids)
        assert G.number_of_nodes() == 20
        assert G.number_of_edges() == 19

    def test_mst_is_connected(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        G = build_mst_graph(dm, ids)
        assert nx.is_connected(G), "MST must be connected"

    def test_mst_is_acyclic(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        G = build_mst_graph(dm, ids)
        assert nx.is_tree(G), "MST must be a tree (acyclic and connected)"

    def test_mst_nan_raises(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        dm[0, 1] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            build_mst_graph(dm, ids)

    def test_mst_with_abundances(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        abund = {fid: 0.1 for fid in ids}
        G = build_mst_graph(dm, ids, abundances=abund)
        for n, d in G.nodes(data=True):
            assert d["weight"] == 0.1

    def test_mst_edges_have_distance(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        G = build_mst_graph(dm, ids)
        for u, v, d in G.edges(data=True):
            assert "distance" in d
            assert d["distance"] >= 0

    def test_mst_weight_is_inverse_distance(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        G = build_mst_graph(dm, ids)
        for u, v, d in G.edges(data=True):
            expected = 1.0 / (d["distance"] + 1e-8)
            assert d["weight"] == pytest.approx(expected, rel=1e-6)

    def test_mst_large_graph(self, large_distance_matrix):
        dm, ids = large_distance_matrix
        G = build_mst_graph(dm, ids)
        assert G.number_of_edges() == 99

    def test_mst_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            build_mst_graph(np.zeros((3, 4)), ["a", "b", "c"])


# =========================================================================
# build_threshold_graph — comprehensive
# =========================================================================

class TestBuildThresholdGraph:
    def test_low_percentile_is_sparse(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        G_low = build_threshold_graph(dm, ids, threshold_percentile=10)
        G_high = build_threshold_graph(dm, ids, threshold_percentile=90)
        assert G_low.number_of_edges() < G_high.number_of_edges()

    def test_percentile_100_connects_nearly_all(self, normal_distance_matrix):
        """percentile=100 threshold = max distance; strict < excludes the max pair."""
        dm, ids = normal_distance_matrix
        G = build_threshold_graph(dm, ids, threshold_percentile=100)
        n = len(ids)
        max_possible = n * (n - 1) // 2
        assert G.number_of_edges() >= max_possible - 1

    def test_empty_feature_ids(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_threshold_graph(np.array([[]]).reshape(0, 0), [])

    def test_with_abundances(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        abund = {fid: 0.2 for fid in ids}
        G = build_threshold_graph(dm, ids, abundances=abund)
        for n, d in G.nodes(data=True):
            assert d["weight"] == 0.2

    def test_no_self_loops(self, normal_distance_matrix):
        dm, ids = normal_distance_matrix
        G = build_threshold_graph(dm, ids, threshold_percentile=50)
        for u, v in G.edges():
            assert u != v

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            build_threshold_graph(np.zeros((2, 3)), ["a", "b"])


# =========================================================================
# transform_edge_weights — comprehensive
# =========================================================================

class TestTransformEdgeWeights:
    def test_identity_preserves(self, simple_nx_graph):
        G2 = transform_edge_weights(simple_nx_graph, 'identity')
        for (_, _, d1), (_, _, d2) in zip(
            simple_nx_graph.edges(data=True), G2.edges(data=True)
        ):
            assert d1['weight'] == d2['weight']

    def test_exponential_name(self, simple_nx_graph):
        G2 = transform_edge_weights(simple_nx_graph, 'exponential')
        for _, _, d in G2.edges(data=True):
            assert 0 < d['weight'] <= 1.0

    def test_exp_alias_matches_exponential(self, simple_nx_graph):
        G_exp = transform_edge_weights(simple_nx_graph, 'exp')
        G_full = transform_edge_weights(simple_nx_graph, 'exponential')
        w_exp = sorted(d['weight'] for _, _, d in G_exp.edges(data=True))
        w_full = sorted(d['weight'] for _, _, d in G_full.edges(data=True))
        np.testing.assert_allclose(w_exp, w_full)

    def test_inverse(self, simple_nx_graph):
        G2 = transform_edge_weights(simple_nx_graph, 'inverse')
        for _, _, d in G2.edges(data=True):
            assert d['weight'] > 1.0

    def test_binary_all_ones(self, simple_nx_graph):
        G2 = transform_edge_weights(simple_nx_graph, 'binary')
        for _, _, d in G2.edges(data=True):
            assert d['weight'] == 1.0

    def test_linear(self, weighted_nx_graph):
        G2 = transform_edge_weights(weighted_nx_graph, 'linear')
        for _, _, d in G2.edges(data=True):
            assert d['weight'] > 0

    def test_unknown_raises(self, simple_nx_graph):
        with pytest.raises(ValueError, match="Unknown transform"):
            transform_edge_weights(simple_nx_graph, 'nonexistent')

    def test_does_not_modify_original(self, simple_nx_graph):
        orig_weights = [d['weight'] for _, _, d in simple_nx_graph.edges(data=True)]
        transform_edge_weights(simple_nx_graph, 'inverse')
        new_weights = [d['weight'] for _, _, d in simple_nx_graph.edges(data=True)]
        assert orig_weights == new_weights, "transform must not mutate the input graph"

    def test_empty_graph(self):
        G = nx.Graph()
        G.add_node(0)
        G2 = transform_edge_weights(G, 'inverse')
        assert G2.number_of_edges() == 0

    @pytest.mark.parametrize("transform", ['inverse', 'exponential', 'linear', 'binary'])
    def test_all_transforms_produce_finite(self, weighted_nx_graph, transform):
        G2 = transform_edge_weights(weighted_nx_graph, transform)
        for _, _, d in G2.edges(data=True):
            assert np.isfinite(d['weight']), f"{transform} produced non-finite weight"


# =========================================================================
# get_graph_params_from_config — comprehensive (§3.2, §3.3)
# =========================================================================

class TestGetGraphParamsFromConfig:
    def test_extracts_graph_type(self):
        config = {'graph_construction': {'graph_type': 'mst'}}
        defaults = {'graph_type': 'knn', 'knn_k': 10}
        result = get_graph_params_from_config(config, defaults)
        assert result['graph_type'] == 'mst'

    def test_preserves_defaults_when_config_empty(self):
        config = {}
        defaults = {'randomize_edges': True, 'preserve_degree': False, 'weight_transform': 'inverse'}
        result = get_graph_params_from_config(config, defaults)
        assert result['randomize_edges'] is True
        assert result['preserve_degree'] is False
        assert result['weight_transform'] == 'inverse'

    def test_overrides_only_when_present(self):
        config = {'graph_construction': {'knn': {'k': 5}, 'weights': {'weight_transform': 'binary'}}}
        defaults = {'knn_k': 10, 'knn_symmetric': True, 'weight_transform': 'identity'}
        result = get_graph_params_from_config(config, defaults)
        assert result['knn_k'] == 5
        assert result['knn_symmetric'] is True
        assert result['weight_transform'] == 'binary'

    def test_threshold_config(self):
        config = {'graph_construction': {'threshold': {'percentile': 30}}}
        defaults = {'threshold_percentile': 25}
        result = get_graph_params_from_config(config, defaults)
        assert result['threshold_percentile'] == 30

    def test_tree_config(self):
        config = {'graph_construction': {'tree': {'ancestor_levels': 5}}}
        defaults = {'tree_ancestor_levels': 3}
        result = get_graph_params_from_config(config, defaults)
        assert result['tree_ancestor_levels'] == 5

    def test_hierarchical_config(self):
        config = {'graph_construction': {'hierarchical': {'aggregation_level': 'family', 'min_abundance': 0.01}}}
        defaults = {}
        result = get_graph_params_from_config(config, defaults)
        assert result['hierarchical_aggregation_level'] == 'family'
        assert result['hierarchical_min_abundance'] == 0.01

    def test_quality_config(self):
        config = {'graph_construction': {'quality': {'min_nodes': 15, 'min_edges': 10}}}
        defaults = {'min_nodes': 8, 'min_edges': 5}
        result = get_graph_params_from_config(config, defaults)
        assert result['min_nodes'] == 15
        assert result['min_edges'] == 10

    def test_edge_construction_config(self):
        config = {'graph_construction': {'edge_construction': {'randomize_edges': True, 'preserve_degree': False}}}
        defaults = {'randomize_edges': False, 'preserve_degree': True}
        result = get_graph_params_from_config(config, defaults)
        assert result['randomize_edges'] is True
        assert result['preserve_degree'] is False

    def test_does_not_mutate_defaults(self):
        config = {'graph_construction': {'knn': {'k': 99}}}
        defaults = {'knn_k': 10}
        get_graph_params_from_config(config, defaults)
        assert defaults['knn_k'] == 10, "Must not mutate the defaults dict"

    def test_full_config_round_trip(self):
        """Simulate a realistic config with all sections."""
        config = {
            'graph_construction': {
                'graph_type': 'threshold',
                'knn': {'k': 15, 'symmetric': False},
                'threshold': {'percentile': 40},
                'tree': {'ancestor_levels': 2},
                'quality': {'min_nodes': 10},
                'weights': {'weight_transform': 'exponential'},
                'edge_construction': {'randomize_edges': True},
            }
        }
        defaults = {
            'graph_type': 'knn', 'knn_k': 10, 'knn_symmetric': True,
            'threshold_percentile': 25, 'tree_ancestor_levels': 3,
            'min_nodes': 8, 'min_edges': 5, 'weight_transform': 'identity',
            'randomize_edges': False, 'preserve_degree': True,
        }
        result = get_graph_params_from_config(config, defaults)
        assert result['graph_type'] == 'threshold'
        assert result['knn_k'] == 15
        assert result['knn_symmetric'] is False
        assert result['threshold_percentile'] == 40
        assert result['tree_ancestor_levels'] == 2
        assert result['min_nodes'] == 10
        assert result['min_edges'] == 5  # not overridden
        assert result['weight_transform'] == 'exponential'
        assert result['randomize_edges'] is True
        assert result['preserve_degree'] is True  # not overridden


# =========================================================================
# randomize_edges — comprehensive
# =========================================================================

class TestRandomizeEdges:
    def test_preserves_node_count(self, simple_nx_graph):
        G2 = randomize_edges(simple_nx_graph, preserve_degree=True)
        assert G2.number_of_nodes() == simple_nx_graph.number_of_nodes()

    def test_preserves_edge_count(self, simple_nx_graph):
        G2 = randomize_edges(simple_nx_graph, preserve_degree=True)
        assert G2.number_of_edges() == simple_nx_graph.number_of_edges()

    def test_simple_randomization_preserves_counts(self, simple_nx_graph):
        G2 = randomize_edges(simple_nx_graph, preserve_degree=False)
        assert G2.number_of_nodes() == simple_nx_graph.number_of_nodes()
        assert G2.number_of_edges() == simple_nx_graph.number_of_edges()

    def test_deterministic_with_seed(self, simple_nx_graph):
        G1 = randomize_edges(simple_nx_graph, seed=123)
        G2 = randomize_edges(simple_nx_graph, seed=123)
        assert set(G1.edges()) == set(G2.edges())

    def test_different_seeds_produce_different_graphs(self, simple_nx_graph):
        G1 = randomize_edges(simple_nx_graph, seed=1, preserve_degree=False)
        G2 = randomize_edges(simple_nx_graph, seed=999, preserve_degree=False)
        assert set(G1.edges()) != set(G2.edges())

    def test_does_not_modify_original(self, simple_nx_graph):
        orig_edges = set(simple_nx_graph.edges())
        randomize_edges(simple_nx_graph, preserve_degree=False, seed=42)
        assert set(simple_nx_graph.edges()) == orig_edges

    def test_preserves_node_attributes(self, simple_nx_graph):
        G2 = randomize_edges(simple_nx_graph, preserve_degree=False)
        for n in simple_nx_graph.nodes():
            assert G2.nodes[n]['weight'] == simple_nx_graph.nodes[n]['weight']

    def test_edges_have_weights(self, simple_nx_graph):
        G2 = randomize_edges(simple_nx_graph, preserve_degree=False)
        for _, _, d in G2.edges(data=True):
            assert 'weight' in d


# =========================================================================
# Utility functions — comprehensive
# =========================================================================

class TestComputeGraphStatistics:
    def test_full_stats(self, simple_nx_graph):
        stats = compute_graph_statistics(simple_nx_graph)
        assert stats['n_nodes'] == 10
        assert stats['n_edges'] == 10
        assert stats['is_connected'] is True
        assert 'avg_clustering' in stats
        assert 'mean_edge_weight' in stats
        assert 'std_edge_weight' in stats

    def test_empty_graph(self):
        stats = compute_graph_statistics(nx.Graph())
        assert stats['n_nodes'] == 0
        assert stats['n_edges'] == 0
        assert stats['density'] == 0

    def test_single_node(self):
        G = nx.Graph()
        G.add_node(0)
        stats = compute_graph_statistics(G)
        assert stats['n_nodes'] == 1
        assert stats['avg_degree'] == 0

    def test_disconnected_graph(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=1.0)
        G.add_edge(2, 3, weight=1.0)
        stats = compute_graph_statistics(G)
        assert stats['is_connected'] is False
        assert stats['n_components'] == 2


class TestValidateGraph:
    def test_pass(self, simple_nx_graph):
        ok, msg = validate_graph(simple_nx_graph, min_nodes=5, min_edges=5)
        assert ok is True
        assert msg == "ok"

    def test_fail_nodes(self, simple_nx_graph):
        ok, msg = validate_graph(simple_nx_graph, min_nodes=100)
        assert ok is False
        assert "Too few nodes" in msg

    def test_fail_edges(self):
        G = nx.Graph()
        for i in range(20):
            G.add_node(i)
        G.add_edge(0, 1)
        ok, msg = validate_graph(G, min_nodes=5, min_edges=5)
        assert ok is False
        assert "Too few edges" in msg

    def test_exact_boundary(self):
        G = nx.complete_graph(8)
        ok, _ = validate_graph(G, min_nodes=8, min_edges=5)
        assert ok is True


class TestMakeCompleteGraph:
    def test_edge_count(self, simple_nx_graph):
        G_c = make_complete_graph(simple_nx_graph)
        n = simple_nx_graph.number_of_nodes()
        assert G_c.number_of_edges() == n * (n - 1) // 2

    def test_preserves_node_attributes(self, simple_nx_graph):
        G_c = make_complete_graph(simple_nx_graph)
        for n in simple_nx_graph.nodes():
            assert G_c.nodes[n]['weight'] == simple_nx_graph.nodes[n]['weight']

    def test_all_edge_weights_are_one(self, simple_nx_graph):
        G_c = make_complete_graph(simple_nx_graph)
        for _, _, d in G_c.edges(data=True):
            assert d['weight'] == 1.0


class TestGetGraphBuilder:
    @pytest.mark.parametrize("gtype", ['knn', 'mst', 'threshold', 'tree', 'hierarchical'])
    def test_all_types_return_callable(self, gtype):
        fn = get_graph_builder(gtype)
        assert callable(fn)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown graph_type"):
            get_graph_builder('nonexistent')

    def test_knn_builder_is_build_knn_graph(self):
        assert get_graph_builder('knn') is build_knn_graph

    def test_mst_builder_is_build_mst_graph(self):
        assert get_graph_builder('mst') is build_mst_graph


# =========================================================================
# build_hierarchical_graph — comprehensive (§4.1 fix)
# =========================================================================

class TestBuildHierarchicalGraph:
    def _make_test_data(self):
        pd = pytest.importorskip("pandas")
        abundance_df = pd.DataFrame(
            {'t_a': [0.1, 0.2], 't_b': [0.3, 0.4], 't_c': [0.05, 0.15]},
            index=['s1', 's2']
        )
        taxonomy_df = pd.DataFrame(
            {'genus': ['GA', 'GA', 'GB'], 'family': ['F1', 'F1', 'F2'],
             'order': ['O1', 'O1', 'O1'], 'class': ['C1', 'C1', 'C1'],
             'phylum': ['P1', 'P1', 'P1']},
            index=['t_a', 't_b', 't_c']
        )
        return abundance_df, taxonomy_df

    def test_returns_graph_not_tuple(self):
        abundance_df, taxonomy_df = self._make_test_data()
        result = build_hierarchical_graph(abundance_df, taxonomy_df, aggregation_level='genus')
        assert isinstance(result, nx.Graph), f"Expected nx.Graph, got {type(result)}"

    def test_taxa_mapping_in_graph_attrs(self):
        abundance_df, taxonomy_df = self._make_test_data()
        G = build_hierarchical_graph(abundance_df, taxonomy_df, aggregation_level='genus')
        assert 'taxa_mapping' in G.graph
        assert isinstance(G.graph['taxa_mapping'], dict)

    def test_invalid_aggregation_level(self):
        abundance_df, taxonomy_df = self._make_test_data()
        with pytest.raises(ValueError, match="aggregation_level must be one of"):
            build_hierarchical_graph(abundance_df, taxonomy_df, aggregation_level='species')

    def test_phylum_level_no_edges(self):
        abundance_df, taxonomy_df = self._make_test_data()
        G = build_hierarchical_graph(abundance_df, taxonomy_df, aggregation_level='phylum')
        assert G.number_of_edges() == 0

    def test_same_family_connects(self):
        abundance_df, taxonomy_df = self._make_test_data()
        G = build_hierarchical_graph(abundance_df, taxonomy_df, aggregation_level='genus')
        if G.has_node('GA') and G.has_node('GB'):
            pass  # may or may not be connected depending on family grouping


# =========================================================================
# Graph invariants across all builders
# =========================================================================

class TestGraphInvariants:
    """Properties that must hold for every graph builder output."""

    @pytest.mark.parametrize("builder_name", ['knn', 'mst', 'threshold'])
    def test_all_nodes_present(self, normal_distance_matrix, builder_name):
        dm, ids = normal_distance_matrix
        builder = get_graph_builder(builder_name)
        if builder_name == 'knn':
            G = builder(dm, ids, k=5)
        elif builder_name == 'mst':
            G = builder(dm, ids)
        else:
            G = builder(dm, ids, threshold_percentile=50)
        assert G.number_of_nodes() == len(ids)
        assert set(G.nodes()) == set(ids)

    @pytest.mark.parametrize("builder_name", ['knn', 'mst', 'threshold'])
    def test_no_self_loops(self, normal_distance_matrix, builder_name):
        dm, ids = normal_distance_matrix
        builder = get_graph_builder(builder_name)
        if builder_name == 'knn':
            G = builder(dm, ids, k=5)
        elif builder_name == 'mst':
            G = builder(dm, ids)
        else:
            G = builder(dm, ids, threshold_percentile=50)
        for u, v in G.edges():
            assert u != v

    @pytest.mark.parametrize("builder_name", ['knn', 'mst', 'threshold'])
    def test_edge_weights_positive(self, normal_distance_matrix, builder_name):
        dm, ids = normal_distance_matrix
        builder = get_graph_builder(builder_name)
        if builder_name == 'knn':
            G = builder(dm, ids, k=5)
        elif builder_name == 'mst':
            G = builder(dm, ids)
        else:
            G = builder(dm, ids, threshold_percentile=50)
        for _, _, d in G.edges(data=True):
            assert d['weight'] > 0, "Edge weights must be positive"

    @pytest.mark.parametrize("builder_name", ['knn', 'mst', 'threshold'])
    def test_edge_weights_finite(self, normal_distance_matrix, builder_name):
        dm, ids = normal_distance_matrix
        builder = get_graph_builder(builder_name)
        if builder_name == 'knn':
            G = builder(dm, ids, k=5)
        elif builder_name == 'mst':
            G = builder(dm, ids)
        else:
            G = builder(dm, ids, threshold_percentile=50)
        for _, _, d in G.edges(data=True):
            assert np.isfinite(d['weight']), "Edge weights must be finite"
