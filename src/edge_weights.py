
import numpy as np
from typing import Callable, Dict


def identity(distance: float, abundance1: float = None, abundance2: float = None) -> float:
    """Raw distance as weight. w = d"""
    return distance


def inverse(distance: float, abundance1: float = None, abundance2: float = None) -> float:
    """Inverse distance - closer means stronger. w = 1/d"""
    return 1.0 / (distance + 1e-8)


def exponential(distance: float, abundance1: float = None, abundance2: float = None) -> float:
    """Exponential decay. w = exp(-d)"""
    return np.exp(-distance)


def binary(distance: float, abundance1: float = None, abundance2: float = None) -> float:
    """Binary - just topology, ignore distance. w = 1"""
    return 1.0


def abundance_product(distance: float, abundance1: float, abundance2: float) -> float:
    inv_dist = 1.0 / (distance + 1e-8)
    return inv_dist * abundance1 * abundance2


def abundance_geometric(distance: float, abundance1: float, abundance2: float) -> float:
    """Geometric mean of abundances. w = (1/d) * sqrt(a1*a2)"""
    inv_dist = 1.0 / (distance + 1e-8)
    return inv_dist * np.sqrt(abundance1 * abundance2 + 1e-8)


def abundance_log(distance: float, abundance1: float, abundance2: float) -> float:
    """Log-transformed. w = (1/d) * log(1+a1) * log(1+a2)"""
    inv_dist = 1.0 / (distance + 1e-8)
    return inv_dist * np.log1p(abundance1) * np.log1p(abundance2)


def abundance_min(distance: float, abundance1: float, abundance2: float) -> float:
    """Min abundance. w = (1/d) * min(a1,a2)"""
    inv_dist = 1.0 / (distance + 1e-8)
    return inv_dist * min(abundance1, abundance2)


def abundance_max(distance: float, abundance1: float, abundance2: float) -> float:
    """Max abundance. w = (1/d) * max(a1,a2)"""
    inv_dist = 1.0 / (distance + 1e-8)
    return inv_dist * max(abundance1, abundance2)


EDGE_WEIGHT_STRATEGIES: Dict[str, Callable] = {
    'identity': identity,
    'inverse': inverse,
    'exponential': exponential,
    'binary': binary,
    'abundance_product': abundance_product,
    'abundance_geometric': abundance_geometric,
    'abundance_log': abundance_log,
    'abundance_min': abundance_min,
    'abundance_max': abundance_max,
}


def get_edge_weight_function(strategy: str) -> Callable:
    if strategy not in EDGE_WEIGHT_STRATEGIES:
        raise ValueError(f"Unknown: {strategy}. Available: {list(EDGE_WEIGHT_STRATEGIES.keys())}")
    return EDGE_WEIGHT_STRATEGIES[strategy]


def compute_edge_weights_for_graph(edges: list, distances: dict, abundances: dict,
                                   strategy: str = 'inverse') -> dict:
    weight_fn = get_edge_weight_function(strategy)
    weights = {}
    needs_abundance = strategy.startswith('abundance_')
    for n1, n2 in edges:
        dist = distances.get((n1, n2)) or distances.get((n2, n1), 1.0)
        if needs_abundance:
            a1 = abundances.get(n1, 0.0)
            a2 = abundances.get(n2, 0.0)
            weights[(n1, n2)] = weight_fn(dist, a1, a2)
        else:
            weights[(n1, n2)] = weight_fn(dist)
    return weights


def list_strategies() -> list:
    return list(EDGE_WEIGHT_STRATEGIES.keys())


if __name__ == "__main__":
    print("Testing edge weight strategies...")
    d = 0.5
    a1, a2 = 0.1, 0.05
    for name, func in EDGE_WEIGHT_STRATEGIES.items():
        if name.startswith('abundance_'):
            w = func(d, a1, a2)
        else:
            w = func(d)
        print(f"{name:25} -> {w:.6f}")
