#!/usr/bin/env python3
"""
Graph utilities for IBD research.
Edge randomization, weight transforms, stats.
"""

import numpy as np
import networkx as nx
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def randomize_edges(G: nx.Graph, preserve_degree: bool = True, seed: int = 42) -> nx.Graph:
    """
    Randomize edges in graph - control experiment to test if specific
    phylogenetic relationships matter or just having graph structure.
    
    If preserve_degree=True, uses double-edge swap to keep degree distribution.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    G_random = G.copy()
    
    if preserve_degree:
        n_swaps = G.number_of_edges() * 10
        try:
            try:
                nx.connected_double_edge_swap(G_random, nswap=n_swaps, seed=seed)
            except (nx.NetworkXError, nx.NetworkXAlgorithmError):
                # fallback
                try:
                    nx.double_edge_swap(G_random, nswap=n_swaps, max_tries=n_swaps*10, seed=seed)
                except (nx.NetworkXError, nx.NetworkXAlgorithmError):
                    print("Warning: swap failed, using simple randomization")
                    G_random = _simple_edge_randomization(G, seed)
        except Exception as e:
            print(f"Warning: randomization failed ({e})")
            G_random = _simple_edge_randomization(G, seed)
    else:
        G_random = _simple_edge_randomization(G, seed)
    
    return G_random


def _simple_edge_randomization(G: nx.Graph, seed: int = 42) -> nx.Graph:
    """Simple randomization - shuffle which nodes connect. Doesnt preserve degrees."""
    random.seed(seed)
    
    nodes = list(G.nodes())
    n_edges = G.number_of_edges()
    
    original_weights = [d.get('weight', 1.0) for _, _, d in G.edges(data=True)]
    random.shuffle(original_weights)
    
    G_random = nx.Graph()
    G_random.add_nodes_from(G.nodes(data=True))
    
    possible_edges = [(u, v) for u in nodes for v in nodes if u < v]
    random.shuffle(possible_edges)
    
    for i, (u, v) in enumerate(possible_edges[:n_edges]):
        weight = original_weights[i] if i < len(original_weights) else 1.0
        G_random.add_edge(u, v, weight=weight)
    
    return G_random


def transform_edge_weights(G: nx.Graph, transform: str = 'inverse') -> nx.Graph:
    """
    Transform edge weights.
    Options: inverse, exp, linear, binary, identity
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
        elif transform == 'exp':
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
        except:
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
    from pathlib import Path
    
    exp_config_path = os.environ.get('EXPERIMENT_CONFIG_PATH')
    
    if exp_config_path and Path(exp_config_path).exists():
        with open(exp_config_path) as f:
            return yaml.safe_load(f)
    
    return {}


def get_graph_params_from_config(experiment_config: Dict, default_params: Dict) -> Dict:
    """Merge experiment config with default graph params."""
    params = default_params.copy()
    
    gc_config = experiment_config.get('graph_construction', {})
    
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
    params['randomize_edges'] = edge_config.get('randomize_edges', False)
    params['preserve_degree'] = edge_config.get('preserve_degree', True)
    
    # weights
    weight_config = gc_config.get('weights', {})
    params['weight_transform'] = weight_config.get('weight_transform', 'identity')
    
    return params


if __name__ == "__main__":
    print("Testing graph_utils...")
    
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
    
    print("\nDone")
