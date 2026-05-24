"""Shared pipeline helpers used by both the notebooks and the experiment runner.

Kept dependency-free (stdlib only) so it can be imported from notebooks, ``src``
modules, and ``scripts/run_experiments.py`` without pulling in heavy packages.
"""
import hashlib
import json
from typing import Any, Dict


def graph_cache_key(config: Dict[str, Any]) -> str:
    """Short hash identifying the graph produced for a given experiment config.

    The per-sample graphs depend on the graph-construction settings (graph type,
    k-NN parameters, edge weight transform, ...) and on which distance matrix is
    primary. Experiments that share a disease but differ in these settings (e.g.
    ``k_5`` vs ``baseline``, ``edge_binary`` vs ``edge_identity``) must NOT reuse
    each other's cached graphs. Keying the graph files by this hash keeps the
    extraction shared per disease while rebuilding graphs whenever the relevant
    settings change. Experiments that only vary the model architecture or training
    hyper-parameters produce the same key and correctly reuse cached graphs.
    """
    graph_construction = config.get('graph_construction', {})
    primary_matrix = (
        config.get('data_extraction', {})
        .get('distance_matrices', {})
        .get('primary_matrix', '')
    )
    payload = json.dumps(
        {'graph_construction': graph_construction, 'primary_matrix': primary_matrix},
        sort_keys=True, default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:8]
