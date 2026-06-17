#!/usr/bin/env python3
"""Paired significance testing across the experiment families.

Turns the n~=50 multi-run results into defensible statistics for the paper.

For every pairwise comparison we pair runs ON SEED (per_run_results carries the
seed), so the two models are compared on identical train/val/test partitions --
a paired design that is far more powerful than comparing independent means.
Reports, per comparison: paired t-test, Wilcoxon signed-rank, paired Cohen's d,
and Benjamini-Hochberg FDR-adjusted p within each comparison family.

Metrics: test AUC-ROC (primary) and balanced accuracy (robust to class imbalance).
Run on the server: ~/BiomeML/.pixi/envs/default/bin/python scripts/run_stats.py
"""
import json
import glob
import os
from pathlib import Path

import numpy as np
from scipy import stats

EXPDIR = Path("experiments")
AUC_KEY = "test_auc_roc"
BACC_KEY = "test_balanced_accuracy"

# The 50 runs of an experiment are repeated random subsamplings of one sample pool
# (single 0.15 hold-out per seed, NOT disjoint folds), so per-seed differences are
# positively correlated. A naive paired t-test treats them as independent and
# overstates significance by ~orders of magnitude. We report the Nadeau-Bengio
# corrected resampled t-test as the primary p-value: variance is inflated by
# (1/n + rho) with rho = test_fraction / (1 - test_fraction). The naive p is kept
# alongside for transparency.
TEST_FRACTION = 0.15
RHO = TEST_FRACTION / (1.0 - TEST_FRACTION)


def load_runs():
    """Return {experiment_name: {seed: {'auc': x, 'bacc': y}}} for all experiments."""
    out = {}
    for agg in sorted(glob.glob(str(EXPDIR / "*" / "aggregated_results.json"))):
        name = Path(agg).parent.name
        try:
            r = json.load(open(agg))
        except Exception:
            continue
        prr = r.get("per_run_results")
        if not prr:
            continue
        runs = {}
        for e in prr:
            seed = e.get("seed")
            m = e.get("metrics", {})
            auc = m.get(AUC_KEY)
            bacc = m.get(BACC_KEY)
            if seed is None or auc is None:
                continue
            runs[seed] = {"auc": float(auc), "bacc": float(bacc) if bacc is not None else np.nan}
        if runs:
            out[name] = runs
    return out


def paired_vectors(runs, a, b, metric):
    """Aligned arrays of metric for experiments a, b over their common seeds."""
    if a not in runs or b not in runs:
        return None, None, None
    seeds = sorted(set(runs[a]) & set(runs[b]))
    if len(seeds) < 5:
        return None, None, None
    va = np.array([runs[a][s][metric] for s in seeds])
    vb = np.array([runs[b][s][metric] for s in seeds])
    return va, vb, seeds


def cohen_d_paired(diff):
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float("nan")


def bh_fdr(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity from the back
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    return adj


def compare(runs, a, b, metric):
    va, vb, seeds = paired_vectors(runs, a, b, metric)
    if va is None:
        return None
    diff = va - vb
    n = len(seeds)
    t_stat, t_p = stats.ttest_rel(va, vb)
    # Nadeau-Bengio corrected resampled t-test (the p-value we actually report).
    sd = diff.std(ddof=1)
    if sd > 0:
        t_nb = diff.mean() / (sd * np.sqrt(1.0 / n + RHO))
        nb_p = float(2 * stats.t.sf(abs(t_nb), df=n - 1))
    else:
        nb_p = 1.0
    try:
        # wilcoxon errors if all diffs zero; guard it
        if np.allclose(diff, 0):
            w_p = 1.0
        else:
            _, w_p = stats.wilcoxon(va, vb)
    except ValueError:
        w_p = float("nan")
    return {
        "a": a, "b": b, "metric": metric, "n_pairs": n,
        "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
        "mean_diff": float(diff.mean()), "naive_t_p": float(t_p), "nb_p": nb_p,
        "w_p": float(w_p), "cohen_d": cohen_d_paired(diff),
    }


# (family, treatment a, baseline b)  -- mean_diff = mean(a) - mean(b)
COMPARISONS = [
    # --- Q1/Q2/controls: does graph help (AGP IBD) ---
    ("AGP-IBD graph-benefit", "baseline", "no_graph_mlp"),
    ("AGP-IBD graph-benefit", "baseline", "cnn_ibd"),
    ("AGP-IBD graph-benefit", "baseline", "shuffle_labels_ibd"),
    ("AGP-IBD graph-benefit", "baseline", "random_edges_ibd"),
    # --- the load-bearing decomposition: is ANY graph > MLP, and is phylo > random ---
    ("AGP-IBD topology", "random_edges_ibd", "no_graph_mlp"),
    ("AGP-IBD topology", "random_edges_ibd", "shuffle_labels_ibd"),
    ("AGP-IBD topology", "baseline", "random_edges_ibd"),
    # --- replication on CMD IBD ---
    ("CMD-IBD graph-benefit", "cmd_ibd_baseline", "cmd_mlp_ibd"),
    ("CMD-IBD graph-benefit", "cmd_ibd_baseline", "cmd_shuffle_ibd"),
    # --- Q5 architecture (AGP IBD, vs GINEConv baseline) ---
    ("AGP-IBD architecture", "baseline", "gcn_arch"),
    ("AGP-IBD architecture", "baseline", "gat_arch"),
    ("AGP-IBD architecture", "baseline", "graphsage_arch"),
    ("AGP-IBD architecture", "baseline", "edgecentric_arch"),
    # --- distance metric (tree vs sequence vs network) ---
    ("AGP-IBD distance", "baseline", "distance_sequence"),
    ("AGP-IBD distance", "baseline", "distance_graph"),
    # --- Q4 edge weighting (vs identity) ---
    ("AGP-IBD edge-weight", "edge_identity", "edge_binary"),
    ("AGP-IBD edge-weight", "edge_identity", "edge_inverse"),
    ("AGP-IBD edge-weight", "edge_identity", "edge_exponential"),
    ("AGP-IBD edge-weight", "edge_identity", "edge_abundance_product"),
    ("AGP-IBD edge-weight", "edge_identity", "edge_abundance_geometric"),
    ("AGP-IBD edge-weight", "edge_identity", "edge_abundance_log"),
    ("AGP-IBD edge-weight", "edge_identity", "edge_abundance_min"),
    ("AGP-IBD edge-weight", "edge_identity", "edge_abundance_max"),
    # --- k-NN density (vs baseline k=10) ---
    ("AGP-IBD knn", "k_15", "baseline"),
    ("AGP-IBD knn", "k_20", "baseline"),
    ("AGP-IBD knn", "k_5", "baseline"),
    # --- per-disease: is the signal real (baseline vs its shuffle) ---
    ("disease-vs-shuffle", "baseline", "shuffle_labels_ibd"),
    ("disease-vs-shuffle", "baseline_t2d", "shuffle_labels_t2d"),
    ("disease-vs-shuffle", "cancer_baseline", "cancer_shuffle"),
    ("disease-vs-shuffle", "autoimmune_baseline", "autoimmune_shuffle"),
    ("disease-vs-shuffle", "cmd_ibd_baseline", "cmd_shuffle_ibd"),
]


def stars(p):
    if p != p:
        return "?"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def main():
    runs = load_runs()
    print(f"Loaded {len(runs)} experiments with per-run results.\n")

    print("Primary p = Nadeau-Bengio corrected (accounts for across-seed test-set overlap);")
    print("naive_p = uncorrected paired t (overstated, shown for transparency);")
    print(f"FDR = Benjamini-Hochberg on the NB p within each family. test_fraction={TEST_FRACTION}.\n")
    for metric, label in [("auc", "AUC-ROC"), ("bacc", "Balanced accuracy")]:
        print("=" * 104)
        print(f"PAIRED COMPARISONS -- metric: {label}  (mean_diff = treatment - baseline)")
        print("=" * 104)
        # group by family
        fams = {}
        for fam, a, b in COMPARISONS:
            res = compare(runs, a, b, metric)
            fams.setdefault(fam, []).append((a, b, res))
        for fam, rows in fams.items():
            valid = [(a, b, r) for a, b, r in rows if r]
            missing = [(a, b) for a, b, r in rows if not r]
            if valid:
                fdr = bh_fdr([r["nb_p"] for _, _, r in valid])
                print(f"\n--- {fam} ---")
                print(f"  {'comparison':42s} {'n':>3s} {'mean_diff':>10s} {'d':>6s} "
                      f"{'nb_p':>9s} {'nb_fdr':>9s} {'naive_p':>9s}  sig")
                for (a, b, r), fp in zip(valid, fdr):
                    comp = f"{a} - {b}"
                    print(f"  {comp:42s} {r['n_pairs']:3d} {r['mean_diff']:+10.4f} "
                          f"{r['cohen_d']:+6.2f} {r['nb_p']:9.4g} {fp:9.4g} "
                          f"{r['naive_t_p']:9.4g}  {stars(fp)}")
            for a, b in missing:
                print(f"  {a} - {b:34s}  (skipped: experiment missing)")
        print()


if __name__ == "__main__":
    main()
