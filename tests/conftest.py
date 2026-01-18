"""Shared fixtures for BiomeML test suite."""

import sys
import os
import pytest
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ---------------------------------------------------------------------------
# Skip markers for optional dependencies
# ---------------------------------------------------------------------------

try:
    import torch
    from torch_geometric.data import Data, Batch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch / torch_geometric not installed")


# ---------------------------------------------------------------------------
# Reusable fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_distance_matrix():
    """5×5 symmetric distance matrix with one zero-distance pair."""
    dm = np.array([
        [0.0, 0.0, 0.3, 0.7, 0.9],
        [0.0, 0.0, 0.4, 0.8, 1.0],
        [0.3, 0.4, 0.0, 0.5, 0.6],
        [0.7, 0.8, 0.5, 0.0, 0.2],
        [0.9, 1.0, 0.6, 0.2, 0.0],
    ])
    feature_ids = ["A", "B", "C", "D", "E"]
    return dm, feature_ids


@pytest.fixture
def normal_distance_matrix():
    """20×20 random symmetric distance matrix (no zero off-diag)."""
    rng = np.random.default_rng(42)
    n = 20
    dm = rng.uniform(0.1, 1.0, (n, n))
    dm = (dm + dm.T) / 2
    np.fill_diagonal(dm, 0)
    ids = [f"taxon_{i}" for i in range(n)]
    return dm, ids


@pytest.fixture
def simple_nx_graph():
    """Small weighted NetworkX graph (10 nodes, ring)."""
    G = nx.cycle_graph(10)
    for u, v in G.edges():
        G[u][v]['weight'] = 0.5
    for n in G.nodes():
        G.nodes[n]['weight'] = 0.1
    return G


@pytest.fixture
def abundances_dict():
    """Abundance dict for 5-node fixture."""
    return {"A": 0.3, "B": 0.1, "C": 0.05, "D": 0.4, "E": 0.15}


@pytest.fixture
def large_distance_matrix():
    """100×100 random symmetric matrix for stress/performance tests."""
    rng = np.random.default_rng(99)
    n = 100
    dm = rng.uniform(0.05, 2.0, (n, n))
    dm = (dm + dm.T) / 2
    np.fill_diagonal(dm, 0)
    ids = [f"ASV_{i:04d}" for i in range(n)]
    return dm, ids


@pytest.fixture
def all_zero_off_diag_matrix():
    """3×3 matrix where all off-diagonal entries are 0 (identical features)."""
    dm = np.zeros((3, 3))
    ids = ["X", "Y", "Z"]
    return dm, ids


@pytest.fixture
def weighted_nx_graph():
    """Graph with varied weights for transform testing."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=0.1)
    G.add_edge(1, 2, weight=0.5)
    G.add_edge(2, 3, weight=1.0)
    G.add_edge(3, 4, weight=2.0)
    G.add_edge(4, 0, weight=0.0)  # zero-weight edge
    for n in G.nodes():
        G.nodes[n]['weight'] = float(n) * 0.1
    return G


if HAS_TORCH:
    @pytest.fixture
    def pyg_batch():
        """Minimal PyG batch with 2 graphs for model testing."""
        x = torch.randn(10, 1)
        edge_index = torch.tensor(
            [[0,1,2,3,4,5,6,7,8,9],
             [1,2,3,4,5,6,7,8,9,0]], dtype=torch.long
        )
        edge_attr = torch.rand(10, 1)
        d1 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(1.0))
        d2 = Data(x=x.clone(), edge_index=edge_index.clone(), edge_attr=edge_attr.clone(), y=torch.tensor(0.0))
        return Batch.from_data_list([d1, d2])

    @pytest.fixture
    def pyg_batch_with_clinical():
        """PyG batch with clinical metadata for GNN_WithMetadata / MetadataOnly."""
        x = torch.randn(10, 1)
        edge_index = torch.tensor(
            [[0,1,2,3,4,5,6,7,8,9],
             [1,2,3,4,5,6,7,8,9,0]], dtype=torch.long
        )
        edge_attr = torch.rand(10, 1)
        clinical = torch.randn(1, 3)
        d1 = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(1.0), clinical=clinical)
        d2 = Data(x=x.clone(), edge_index=edge_index.clone(), edge_attr=edge_attr.clone(),
                  y=torch.tensor(0.0), clinical=clinical.clone())
        return Batch.from_data_list([d1, d2])
