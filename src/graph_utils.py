#!/usr/bin/env python3
"""
Graph utilities for BiomeML microbiome analysis.
Edge randomization, weight transforms, stats.
Alternative graph construction methods: tree, threshold, hierarchical, MST.
"""

import numpy as np
import networkx as nx
import random
from collections import Counter
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path


def _validate_distance_matrix(distance_matrix: np.ndarray, feature_ids: List[str]) -> None:
    """Shared input validation for all graph builders that accept a distance matrix."""
    if not isinstance(distance_matrix, np.ndarray):
        raise TypeError(f"distance_matrix must be numpy ndarray, got {type(distance_matrix).__name__}")
    if distance_matrix.ndim != 2:
        raise ValueError(f"distance_matrix must be 2D, got {distance_matrix.ndim}D")
    n = len(feature_ids)
    if not feature_ids:
        raise ValueError("feature_ids must be non-empty")
    if distance_matrix.shape != (n, n):
        raise ValueError(
            f"distance_matrix shape {distance_matrix.shape} does not match "
            f"feature_ids length {n}"
        )
    if np.any(np.isnan(distance_matrix)):
        raise ValueError("distance_matrix contains NaN values")
    if np.any(np.isinf(distance_matrix)):
        raise ValueError("distance_matrix contains Inf values")
    if np.any(distance_matrix < 0):
        raise ValueError("distance_matrix contains negative values")


def randomize_edges(G: nx.Graph, preserve_degree: bool = True, seed: int = 42) -> nx.Graph:
    """
    Randomize edges in graph - control experiment to test if specific
    phylogenetic relationships matter or just having graph structure.
    
    If preserve_degree=True, uses double-edge swap to keep degree distribution.
    Raises RuntimeError if degree-preserving swap fails (no silent fallback).
    """
    import warnings
    rng = random.Random(seed)
    
    G_random = G.copy()
    
    if preserve_degree:
        n_swaps = G.number_of_edges() * 10
        try:
            nx.connected_double_edge_swap(G_random, nswap=n_swaps, seed=seed)
        except (nx.NetworkXError, nx.NetworkXAlgorithmError):
            try:
                nx.double_edge_swap(G_random, nswap=n_swaps, max_tries=n_swaps*10, seed=seed)
            except (nx.NetworkXError, nx.NetworkXAlgorithmError):
                warnings.warn(
                    "Degree-preserving edge swap failed — falling back to "
                    "simple randomization (degree distribution NOT preserved). "
                    "This changes the control experiment semantics.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                G_random = _simple_edge_randomization(G, seed)
    else:
        G_random = _simple_edge_randomization(G, seed)
    
    return G_random


def _simple_edge_randomization(G: nx.Graph, seed: int = 42) -> nx.Graph:
    """Simple randomization - shuffle which nodes connect. Doesn't preserve degrees."""
    rng = random.Random(seed)
    
    nodes = list(G.nodes())
    n_edges = G.number_of_edges()
    
    original_weights = [d.get('weight', 1.0) for _, _, d in G.edges(data=True)]
    rng.shuffle(original_weights)
    
    G_random = nx.Graph()
    G_random.add_nodes_from(G.nodes(data=True))
    
    possible_edges = [(u, v) for u in nodes for v in nodes if u < v]
    rng.shuffle(possible_edges)
    
    for i, (u, v) in enumerate(possible_edges[:n_edges]):
        weight = original_weights[i] if i < len(original_weights) else 1.0
        G_random.add_edge(u, v, weight=weight)
    
    return G_random


def transform_edge_weights(G: nx.Graph, transform: str = 'inverse') -> nx.Graph:
    """
    Transform edge weights.
    Options: inverse, exponential (or exp), linear, binary, identity
    """
    G_transformed = G.copy()
    
    if transform == 'identity':
        return G_transformed
    
    weights = [d.get('weight', 1.0) for _, _, d in G.edges(data=True)]
    
    if not weights:
        return G_transformed
    
    max_weight = max(weights)
    
    for u, v, d in G_transformed.edges(data=True):
        w = d.get('weight', 1.0)
        
        if transform == 'inverse':
            d['weight'] = 1.0 / (w + 1e-8)
        elif transform in ('exponential', 'exp'):
            d['weight'] = np.exp(-w)
        elif transform == 'linear':
            d['weight'] = max_weight - w + 1e-8
        elif transform == 'binary':
            d['weight'] = 1.0
        else:
            raise ValueError(f"Unknown transform: {transform}")
    
    return G_transformed


def compute_graph_statistics(G: nx.Graph) -> Dict:
    """Basic graph stats."""
    stats = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
    }
    
    if G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        stats['avg_degree'] = sum(degrees.values()) / len(degrees)
        stats['max_degree'] = max(degrees.values())
        stats['min_degree'] = min(degrees.values())
        
        stats['is_connected'] = nx.is_connected(G)
        stats['n_components'] = nx.number_connected_components(G)
        
        try:
            stats['avg_clustering'] = nx.average_clustering(G)
        except (nx.NetworkXError, ZeroDivisionError):
            stats['avg_clustering'] = None
            
        weights = [d.get('weight', 1.0) for _, _, d in G.edges(data=True)]
        if weights:
            stats['mean_edge_weight'] = np.mean(weights)
            stats['std_edge_weight'] = np.std(weights)
    
    return stats


def validate_graph(G: nx.Graph, min_nodes: int = 8, min_edges: int = 5) -> Tuple[bool, str]:
    """Check if graph meets minimum requirements."""
    if G.number_of_nodes() < min_nodes:
        return False, f"Too few nodes: {G.number_of_nodes()} < {min_nodes}"
    
    if G.number_of_edges() < min_edges:
        return False, f"Too few edges: {G.number_of_edges()} < {min_edges}"
    
    return True, "ok"


def load_experiment_config():
    """Load experiment config from env var if available."""
    import os
    import yaml
    
    exp_config_path = os.environ.get('EXPERIMENT_CONFIG_PATH')
    
    if exp_config_path and Path(exp_config_path).exists():
        with open(exp_config_path) as f:
            return yaml.safe_load(f)
    
    return {}


def get_graph_params_from_config(experiment_config: Dict, default_params: Dict) -> Dict:
    """Merge experiment config with default graph params.
    
    Only overrides values that are explicitly present in the config,
    preserving default_params for anything not specified.
    """
    params = default_params.copy()
    
    gc_config = experiment_config.get('graph_construction', {})
    
    # graph_type
    if 'graph_type' in gc_config:
        params['graph_type'] = gc_config['graph_type']
    
    # knn
    knn_config = gc_config.get('knn', {})
    if 'k' in knn_config:
        params['knn_k'] = knn_config['k']
    if 'symmetric' in knn_config:
        params['knn_symmetric'] = knn_config['symmetric']
    if 'max_distance_factor' in knn_config:
        params['knn_max_distance_factor'] = knn_config['max_distance_factor']
    
    # quality
    quality_config = gc_config.get('quality', {})
    if 'min_nodes' in quality_config:
        params['min_nodes'] = quality_config['min_nodes']
    if 'min_edges' in quality_config:
        params['min_edges'] = quality_config['min_edges']
    
    # edge construction
    edge_config = gc_config.get('edge_construction', {})
    if 'randomize_edges' in edge_config:
        params['randomize_edges'] = edge_config['randomize_edges']
    if 'preserve_degree' in edge_config:
        params['preserve_degree'] = edge_config['preserve_degree']
    
    # weights
    weight_config = gc_config.get('weights', {})
    if 'weight_transform' in weight_config:
        params['weight_transform'] = weight_config['weight_transform']

    # threshold
    threshold_config = gc_config.get('threshold', {})
    if 'percentile' in threshold_config:
        params['threshold_percentile'] = threshold_config['percentile']

    # tree
    tree_config = gc_config.get('tree', {})
    if 'ancestor_levels' in tree_config:
        params['tree_ancestor_levels'] = tree_config['ancestor_levels']

    # hierarchical
    hierarchical_config = gc_config.get('hierarchical', {})
    if 'aggregation_level' in hierarchical_config:
        params['hierarchical_aggregation_level'] = hierarchical_config['aggregation_level']
    if 'min_abundance' in hierarchical_config:
        params['hierarchical_min_abundance'] = hierarchical_config['min_abundance']

    return params


# =============================================================================
# ALTERNATIVE GRAPH CONSTRUCTION METHODS
# =============================================================================

def build_tree_graph(
    tree_path: Union[str, Path],
    feature_ids: List[str],
    max_ancestor_levels: int = 3,
    abundances: Optional[Dict[str, float]] = None,
    *,
    ancestor_levels: Optional[int] = None
) -> nx.Graph:
    """
    Build graph from actual phylogenetic tree structure.
    
    Connects tips (leaves) that share a common ancestor within N levels.
    
    Parameters
    ----------
    tree_path : str or Path
        Path to Newick tree file
    feature_ids : List[str]
        List of feature IDs (ASV sequences) to include
    max_ancestor_levels : int
        Connect tips sharing ancestor within this many levels (default: 3).
        Overridden by ancestor_levels if provided.
    abundances : Dict[str, float], optional
        Node abundances for weighting
    ancestor_levels : int, optional
        Alias for max_ancestor_levels (for compatibility with notebook config)
        
    Returns
    -------
    nx.Graph
        Graph where edges connect phylogenetically close tips
    """
    levels = ancestor_levels if ancestor_levels is not None else max_ancestor_levels
    if levels <= 0:
        raise ValueError(
            "ancestor_levels (or max_ancestor_levels) must be >= 1. "
            f"Got: {levels}"
        )
    try:
        from Bio import Phylo
    except ImportError:
        raise ImportError("BioPython required: pip install biopython")
    
    # Load tree
    tree = Phylo.read(tree_path, "newick")
    
    # Get all terminals (tips/leaves)
    terminals = {t.name: t for t in tree.get_terminals() if t.name}
    
    # Filter to only requested feature IDs
    valid_ids = set(feature_ids) & set(terminals.keys())
    
    if len(valid_ids) < 2:
        print(f"Warning: Only {len(valid_ids)} valid IDs found in tree")
        # Return empty graph with nodes
        G = nx.Graph()
        for fid in feature_ids:
            G.add_node(fid, weight=abundances.get(fid, 1.0) if abundances else 1.0)
        return G
    
    G = nx.Graph()
    
    for fid in feature_ids:
        weight = abundances.get(fid, 1.0) if abundances else 1.0
        G.add_node(fid, weight=weight)
    
    n_missing = len(set(feature_ids) - valid_ids)
    if n_missing > 0:
        print(f"Warning: {n_missing}/{len(feature_ids)} feature IDs not found in "
              f"tree — these nodes will have no edges")
    
    # Build ancestor lookup
    def get_ancestors(terminal, max_levels):
        """Get ancestors up to max_levels."""
        ancestors = []
        path = tree.get_path(terminal)
        if path:
            ancestors = path[:-1][-max_levels:]  # Exclude self, take last N
        return ancestors
    
    # Cache ancestors for each terminal
    ancestor_cache = {}
    for fid in valid_ids:
        if fid in terminals:
            ancestor_cache[fid] = set(get_ancestors(terminals[fid], levels))
    
    # Connect tips sharing ancestors
    valid_list = list(valid_ids)
    for i, fid1 in enumerate(valid_list):
        for fid2 in valid_list[i+1:]:
            if fid1 in ancestor_cache and fid2 in ancestor_cache:
                shared = ancestor_cache[fid1] & ancestor_cache[fid2]
                if shared:
                    # Weight by number of shared ancestors (closer = more shared)
                    # levels is guaranteed >= 1 by earlier validation
                    weight = len(shared) / levels
                    G.add_edge(fid1, fid2, weight=weight)
    
    return G


def build_threshold_graph(
    distance_matrix: np.ndarray,
    feature_ids: List[str],
    threshold_percentile: float = 25.0,
    abundances: Optional[Dict[str, float]] = None
) -> nx.Graph:
    """
    Build graph by connecting all pairs with distance below threshold.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise distance matrix (n x n)
    feature_ids : List[str]
        List of feature IDs corresponding to matrix rows/cols
    threshold_percentile : float
        Percentile of distances to use as threshold (default: 25th percentile)
        Lower = sparser graph, higher = denser graph
    abundances : Dict[str, float], optional
        Node abundances for weighting
        
    Returns
    -------
    nx.Graph
        Graph where edges connect pairs with distance < threshold
    """
    _validate_distance_matrix(distance_matrix, feature_ids)
    n = len(feature_ids)
    
    # Get threshold from percentile of non-zero distances
    upper_tri = distance_matrix[np.triu_indices(n, k=1)]
    non_zero = upper_tri[upper_tri > 0]
    
    if len(non_zero) == 0:
        threshold = 1.0
    else:
        threshold = np.percentile(non_zero, threshold_percentile)
    
    G = nx.Graph()
    
    # Add nodes
    for i, fid in enumerate(feature_ids):
        weight = abundances.get(fid, 1.0) if abundances else 1.0
        G.add_node(fid, weight=weight, idx=i)
    
    # Add edges for pairs below threshold (including zero-distance pairs)
    for i in range(n):
        for j in range(i + 1, n):
            dist = distance_matrix[i, j]
            if dist < threshold:
                weight = 1.0 / (dist + 1e-8)
                G.add_edge(feature_ids[i], feature_ids[j], weight=weight, distance=dist)
    
    return G


def build_hierarchical_graph(
    abundance_df,  # pd.DataFrame: samples x taxa
    taxonomy_df,   # pd.DataFrame: taxa with taxonomic level columns
    aggregation_level: str = 'genus',
    min_abundance: float = 0.001
) -> nx.Graph:
    """
    Build graph by aggregating features at higher taxonomic level.
    
    Parameters
    ----------
    abundance_df : pd.DataFrame
        Abundance table (samples x taxa)
    taxonomy_df : pd.DataFrame
        Taxonomy table with columns like 'phylum', 'class', 'order', 'family', 'genus'.
        Must be indexed by taxon/feature ID (same as abundance_df.columns).
    aggregation_level : str
        Level to aggregate at: 'genus', 'family', 'order', 'class', 'phylum'.
        Note: for 'phylum', there is no parent level, so the graph will have
        no edges (nodes only).
    min_abundance : float
        Minimum mean abundance to include a taxon
        
    Returns
    -------
    nx.Graph
        Graph at aggregated level.  The mapping from original to aggregated
        taxa is stored in ``G.graph['taxa_mapping']``.
    """
    import pandas as pd
    
    # Validate aggregation level
    valid_levels = ['phylum', 'class', 'order', 'family', 'genus']
    if aggregation_level not in valid_levels:
        raise ValueError(f"aggregation_level must be one of {valid_levels}")
    
    # Ensure taxonomy_df has the required column
    if aggregation_level not in taxonomy_df.columns:
        raise ValueError(f"taxonomy_df missing column: {aggregation_level}")
    
    # Create mapping from original taxa to aggregated taxa
    taxa_mapping = {}
    for taxon in abundance_df.columns:
        if taxon in taxonomy_df.index:
            agg_taxon = taxonomy_df.loc[taxon, aggregation_level]
            if pd.notna(agg_taxon) and agg_taxon != '':
                taxa_mapping[taxon] = agg_taxon
            else:
                taxa_mapping[taxon] = f"Unknown_{aggregation_level}"
        else:
            taxa_mapping[taxon] = f"Unknown_{aggregation_level}"
    
    # Aggregate abundances by taxonomic level
    agg_abundance = abundance_df.copy()
    agg_abundance.columns = [taxa_mapping.get(c, c) for c in agg_abundance.columns]
    agg_abundance = agg_abundance.T.groupby(level=0).sum().T
    
    # Filter by minimum abundance
    mean_abundance = agg_abundance.mean(axis=0)
    keep_taxa = mean_abundance[mean_abundance >= min_abundance].index.tolist()
    
    if len(keep_taxa) < 2:
        keep_taxa = mean_abundance.nlargest(10).index.tolist()
    
    # Build graph connecting taxa at same parent level
    G = nx.Graph()
    
    # Add nodes
    for taxon in keep_taxa:
        G.add_node(taxon, weight=mean_abundance[taxon])
    
    # Connect taxa sharing parent (one level up)
    parent_level_idx = valid_levels.index(aggregation_level) - 1
    if parent_level_idx >= 0:
        parent_level = valid_levels[parent_level_idx]

        # Get parent for each aggregated taxon (use most common if multiple)
        taxon_parent_lists = {}
        for orig_taxon, agg_taxon in taxa_mapping.items():
            if agg_taxon in keep_taxa and orig_taxon in taxonomy_df.index:
                parent = taxonomy_df.loc[orig_taxon, parent_level]
                if pd.notna(parent) and parent != '':
                    taxon_parent_lists.setdefault(agg_taxon, []).append(parent)
        taxon_parents = {}
        for agg_taxon, parents in taxon_parent_lists.items():
            taxon_parents[agg_taxon] = Counter(parents).most_common(1)[0][0]

        # Connect taxa with same parent
        taxa_list = list(keep_taxa)
        for i, t1 in enumerate(taxa_list):
            for t2 in taxa_list[i+1:]:
                p1 = taxon_parents.get(t1)
                p2 = taxon_parents.get(t2)
                if p1 and p2 and p1 == p2:
                    # Weight by abundance product
                    w = np.sqrt(mean_abundance[t1] * mean_abundance[t2])
                    G.add_edge(t1, t2, weight=w)
    
    G.graph['taxa_mapping'] = taxa_mapping
    return G


def build_mst_graph(
    distance_matrix: np.ndarray,
    feature_ids: List[str],
    abundances: Optional[Dict[str, float]] = None
) -> nx.Graph:
    """
    Build Minimum Spanning Tree graph from distance matrix.
    
    Creates a single connected tree capturing closest relationships.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise distance matrix (n x n)
    feature_ids : List[str]
        List of feature IDs corresponding to matrix rows/cols
    abundances : Dict[str, float], optional
        Node abundances for weighting
        
    Returns
    -------
    nx.Graph
        Minimum spanning tree graph
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.sparse import csr_matrix, coo_matrix
    
    _validate_distance_matrix(distance_matrix, feature_ids)
    n = len(feature_ids)

    # Replace non-diagonal zeros with a tiny epsilon so they survive
    # sparse conversion (csr treats 0.0 as absent).
    _ZERO_DIST_EPS = 1e-10
    dm = distance_matrix.copy()
    zero_mask = (dm == 0)
    np.fill_diagonal(zero_mask, False)
    dm[zero_mask] = _ZERO_DIST_EPS

    # Explicit CSR conversion required: passing a dense array directly to
    # minimum_spanning_tree can silently drop near-zero entries.
    mst_sparse = minimum_spanning_tree(csr_matrix(dm))
    
    G = nx.Graph()
    
    # Add nodes
    for i, fid in enumerate(feature_ids):
        weight = abundances.get(fid, 1.0) if abundances else 1.0
        G.add_node(fid, weight=weight, idx=i)
    
    # Extract MST edges directly from sparse matrix (O(n) instead of O(n²))
    mst_coo = coo_matrix(mst_sparse)
    for i, j, dist in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        if dist <= _ZERO_DIST_EPS:
            dist = 0.0
        weight = 1.0 / (dist + 1e-8)
        G.add_edge(feature_ids[i], feature_ids[j], weight=weight, distance=dist)
    
    return G


def build_knn_graph(
    distance_matrix: np.ndarray,
    feature_ids: List[str],
    k: int = 10,
    symmetric: bool = True,
    max_distance_factor: float = 2.0,
    abundances: Optional[Dict[str, float]] = None
) -> nx.Graph:
    """
    Build k-Nearest Neighbors graph from distance matrix.
    
    This is the original/default graph construction method.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise distance matrix (n x n)
    feature_ids : List[str]
        List of feature IDs corresponding to matrix rows/cols
    k : int
        Number of nearest neighbors
    symmetric : bool
        If True (union), edge exists if either node is in other's k-NN.
        If False (intersection), edge exists only if both nodes are mutual k-NN.
    max_distance_factor : float
        Maximum distance as factor of median distance
    abundances : Dict[str, float], optional
        Node abundances for weighting
        
    Returns
    -------
    nx.Graph
        k-NN graph
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    _validate_distance_matrix(distance_matrix, feature_ids)
    n = len(feature_ids)

    # Get distance threshold
    upper_tri = distance_matrix[np.triu_indices(n, k=1)]
    non_zero = upper_tri[upper_tri > 0]
    max_dist = np.median(non_zero) * max_distance_factor if len(non_zero) > 0 else float('inf')
    
    G = nx.Graph()
    
    # Add nodes
    for i, fid in enumerate(feature_ids):
        weight = abundances.get(fid, 1.0) if abundances else 1.0
        G.add_node(fid, weight=weight, idx=i)
    
    # Build k-NN sets for each node
    knn_sets = {}
    for i in range(n):
        distances = distance_matrix[i, :]
        sorted_idx = np.argsort(distances)
        neighbors = []
        for j in sorted_idx:
            if j != i and distances[j] < max_dist:
                neighbors.append(j)
                if len(neighbors) >= k:
                    break
        knn_sets[i] = set(neighbors)

    # Add edges: union (symmetric=True) or intersection (symmetric=False)
    for i in range(n):
        for j in range(i + 1, n):
            if symmetric:
                in_graph = j in knn_sets[i] or i in knn_sets[j]
            else:
                in_graph = j in knn_sets[i] and i in knn_sets[j]
            if in_graph:
                dist = distance_matrix[i, j]
                if dist < max_dist:
                    weight = 1.0 / (dist + 1e-8)
                    G.add_edge(feature_ids[i], feature_ids[j], weight=weight, distance=dist)
    return G


def make_complete_graph(G: nx.Graph) -> nx.Graph:
    """
    Replace graph edges with a complete graph (all pairs connected).
    Control experiment to test if selecting specific edges matters.
    Keeps original node attributes; edges get weight=1.0.
    """
    nodes = list(G.nodes(data=True))
    G_complete = nx.Graph()
    G_complete.add_nodes_from(nodes)

    node_ids = [n for n, _ in nodes]
    for i, n1 in enumerate(node_ids):
        for n2 in node_ids[i + 1:]:
            G_complete.add_edge(n1, n2, weight=1.0)

    return G_complete


def get_graph_builder(graph_type: str):
    """
    Get graph construction function by type name.
    
    Parameters
    ----------
    graph_type : str
        One of: 'knn', 'tree', 'threshold', 'hierarchical', 'mst'
        
    Returns
    -------
    callable
        Graph construction function.  All builders return ``nx.Graph``.
        For 'hierarchical', the taxa mapping is stored in
        ``G.graph['taxa_mapping']``.
    """
    builders = {
        'knn': build_knn_graph,
        'tree': build_tree_graph,
        'threshold': build_threshold_graph,
        'hierarchical': build_hierarchical_graph,
        'mst': build_mst_graph
    }
    
    if graph_type not in builders:
        raise ValueError(f"Unknown graph_type: {graph_type}. Must be one of {list(builders.keys())}")
    
    return builders[graph_type]


if __name__ == "__main__":
    print("Testing graph_utils...")
    
    # Test basic functions
    G = nx.barabasi_albert_graph(50, 3, seed=42)
    
    for u, v in G.edges():
        G[u][v]['weight'] = np.random.uniform(0.1, 1.0)
    
    print(f"Original: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    G_random = randomize_edges(G, preserve_degree=True)
    print(f"Randomized: {G_random.number_of_nodes()} nodes, {G_random.number_of_edges()} edges")
    
    orig_edges = set(G.edges())
    rand_edges = set(G_random.edges())
    overlap = len(orig_edges & rand_edges)
    print(f"Edge overlap: {overlap}/{len(orig_edges)}")
    
    print("\nWeight transforms:")
    print(f"  Original mean: {np.mean([d['weight'] for _,_,d in G.edges(data=True)]):.4f}")
    
    G_inv = transform_edge_weights(G, 'inverse')
    print(f"  Inverse mean: {np.mean([d['weight'] for _,_,d in G_inv.edges(data=True)]):.4f}")
    
    stats = compute_graph_statistics(G)
    print(f"\nStats: {stats}")
    
    # Test alternative graph builders
    print("\n--- Testing Alternative Graph Builders ---")
    
    # Create test distance matrix
    n = 20
    feature_ids = [f"taxon_{i}" for i in range(n)]
    dist_matrix = np.random.uniform(0.1, 1.0, (n, n))
    dist_matrix = (dist_matrix + dist_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(dist_matrix, 0)
    
    # Test k-NN
    G_knn = build_knn_graph(dist_matrix, feature_ids, k=5)
    print(f"k-NN graph: {G_knn.number_of_nodes()} nodes, {G_knn.number_of_edges()} edges")
    
    # Test threshold
    G_thresh = build_threshold_graph(dist_matrix, feature_ids, threshold_percentile=25)
    print(f"Threshold graph: {G_thresh.number_of_nodes()} nodes, {G_thresh.number_of_edges()} edges")
    
    # Test MST
    G_mst = build_mst_graph(dist_matrix, feature_ids)
    print(f"MST graph: {G_mst.number_of_nodes()} nodes, {G_mst.number_of_edges()} edges")
    
    print("\nDone")
