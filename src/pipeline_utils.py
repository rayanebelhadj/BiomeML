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
    data_extraction = config.get('data_extraction', {})
    dataset = config.get('dataset', {})
    dataset_name = dataset.get('name', '') if isinstance(dataset, dict) else str(dataset)
    # Key on the dataset, the graph-construction settings, and which distance matrix
    # is primary. CRITICAL: these must be STABLE across the runs of one experiment so
    # the multi-run path (run 1 builds graphs; runs 2..N reuse them) finds the cache.
    # Do NOT include matching/feature_filtering here: matching carries a per-run
    # random_seed (injected by run_experiments) that would change the key every run
    # and break graph reuse. Extraction-level settings are scoped by the per-disease
    # extraction directory, not by this graph key.
    payload = json.dumps(
        {
            'dataset': dataset_name,
            'graph_construction': graph_construction,
            'primary_matrix': data_extraction.get('distance_matrices', {}).get('primary_matrix', ''),
        },
        sort_keys=True, default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:8]
