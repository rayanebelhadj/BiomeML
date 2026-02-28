"""Shared helpers for loading experiments, results, and status."""

import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_YAML = PROJECT_ROOT / "experiments.yaml"
CONFIG_YAML = PROJECT_ROOT / "config.yaml"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

DATASETS = {
    "agp": "American Gut Project",
    "cmd": "curatedMetagenomicData",
    "hmp": "Human Microbiome Project",
}

AGP_DISEASES = [
    "ibd", "diabetes", "cancer", "autoimmune", "depression",
    "mental_illness", "ptsd", "arthritis", "asthma", "stomach_bowel",
]
CMD_DISEASES = ["IBD", "CRC", "T2D", "Cirrhosis", "Obesity"]
HMP_DISEASES = ["IBD", "Crohns", "UC"]

CATEGORIES = {
    "arch": ["baseline", "gcn", "gat", "graphsage", "edgecentric", "mlp", "cnn"],
    "control": ["random_edges", "complete_graph", "shuffled_labels"],
    "metadata": ["gnn_meta", "metadata_only"],
    "multiclass": ["age"],
}


def get_diseases_for_dataset(dataset_name: str) -> List[str]:
    if dataset_name == "cmd":
        return CMD_DISEASES
    elif dataset_name == "hmp":
        return HMP_DISEASES
    return AGP_DISEASES


def load_experiments() -> Dict:
    """Load experiments.yaml, returning only dict entries (actual experiments)."""
    try:
        with open(EXPERIMENTS_YAML) as f:
            data = yaml.safe_load(f)
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        return {}


def get_experiment_status(exp_name: str) -> Tuple[int, str]:
    """Return (completed_runs, status_text) for an experiment."""
    exp_dir = EXPERIMENTS_DIR / exp_name
    if not exp_dir.exists():
        return 0, "not started"

    runs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    completed = sum(1 for r in runs if (r / "evaluation_results.json").exists())

    agg_file = exp_dir / "aggregated_results.json"
    if agg_file.exists():
        try:
            with open(agg_file) as f:
                data = json.load(f)
            acc = data.get("test_accuracy", {}).get("mean")
            if acc is None:
                acc = data.get("metrics", {}).get("test_accuracy", {}).get("mean")
            if acc is not None:
                return completed, f"{completed} runs, {acc:.1%} acc"
        except Exception:
            pass

    if completed > 0:
        return completed, f"{completed} runs"
    return 0, "not started"


def get_experiments_for_disease(disease: str, experiments: Dict) -> List[str]:
    results = []
    for name, config in experiments.items():
        if isinstance(config, dict) and config.get("disease") == disease:
            results.append(name)
    return sorted(results)


def get_experiments_for_dataset(dataset_name: str, experiments: Dict) -> List[str]:
    results = []
    for name, config in experiments.items():
        if isinstance(config, dict):
            if config.get("dataset", "agp") == dataset_name:
                results.append(name)
    return sorted(results)


# ---------------------------------------------------------------------------
# Parsing helpers (ported from scripts/analyze_results.py)
# ---------------------------------------------------------------------------

def parse_disease(exp_name: str) -> str:
    for disease in AGP_DISEASES:
        if exp_name.startswith(disease):
            return disease
    if exp_name.startswith("cmd_"):
        parts = exp_name.split("_")
        if len(parts) >= 2:
            return parts[1]
    if exp_name in (
        "baseline", "gnn_gcn", "gnn_gineconv", "gnn_gat",
        "gnn_graphsage", "gnn_edgecentric", "cnn_baseline",
        "no_graph_mlp", "random_edges", "complete_graph",
        "shuffled_labels",
    ):
        return "ibd"
    if exp_name.startswith(("edge_", "k_", "distance_", "hp_", "meta_")):
        return "ibd"
    return "ibd"


def parse_model_type(exp_name: str) -> str:
    if "cnn" in exp_name:
        return "CNN"
    if "mlp" in exp_name:
        return "MLP"
    if "gcn" in exp_name and "gineconv" not in exp_name:
        return "GCN"
    if "gineconv" in exp_name or exp_name == "baseline":
        return "GINEConv"
    if "gat" in exp_name:
        return "GAT"
    if "graphsage" in exp_name:
        return "GraphSAGE"
    if "edgecentric" in exp_name:
        return "EdgeCentricRGCN"
    return "Other"


def parse_category(exp_name: str) -> str:
    if exp_name.startswith("k_"):
        return "k-NN Density"
    if exp_name.startswith("distance_"):
        return "Distance Matrix"
    if exp_name.startswith("edge_"):
        return "Edge Weight"
    if exp_name.startswith("hp_"):
        return "Hyperparameter"
    if exp_name.startswith("meta_"):
        return "Metadata"
    if "metadata_only" in exp_name:
        return "Metadata Only"
    if "gnn_meta" in exp_name:
        return "GNN + Metadata"
    if "random_edges" in exp_name:
        return "Control: Random"
    if "complete_graph" in exp_name:
        return "Control: Complete"
    if "shuffled_labels" in exp_name:
        return "Control: Shuffled"
    if "age" in exp_name and "meta_age" not in exp_name:
        return "Multi-class: Age"
    if "subtypes" in exp_name:
        return "Multi-class: Subtypes"
    if exp_name.endswith("_baseline") or exp_name.startswith("gnn_"):
        return "Architecture"
    if exp_name in ("baseline", "cnn_baseline", "no_graph_mlp"):
        return "Architecture"
    return "Other"


# ---------------------------------------------------------------------------
# Results loading
# ---------------------------------------------------------------------------

def load_all_results() -> pd.DataFrame:
    """Load aggregated_results.json from every experiment directory into a DataFrame."""
    if not EXPERIMENTS_DIR.exists():
        return pd.DataFrame()

    results = []
    for exp_dir in EXPERIMENTS_DIR.iterdir():
        if not exp_dir.is_dir():
            continue
        agg_file = exp_dir / "aggregated_results.json"
        if not agg_file.exists():
            continue
        try:
            with open(agg_file) as f:
                data = json.load(f)
            metrics = data.get("metrics", {})
            result = {
                "experiment": exp_dir.name,
                "num_runs": data.get(
                    "num_runs",
                    len(metrics.get("test_accuracy", {}).get("values", [])),
                ),
            }
            for metric_name in (
                "test_accuracy", "test_auc", "test_f1",
                "test_precision", "test_recall", "test_balanced_accuracy",
            ):
                if metric_name in metrics:
                    result[f"{metric_name}_mean"] = metrics[metric_name].get("mean")
                    result[f"{metric_name}_std"] = metrics[metric_name].get("std")
                    result[f"{metric_name}_ci_lower"] = metrics[metric_name].get("ci_lower")
                    result[f"{metric_name}_ci_upper"] = metrics[metric_name].get("ci_upper")
            results.append(result)
        except Exception:
            continue

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df["disease"] = df["experiment"].apply(parse_disease)
    df["model_type"] = df["experiment"].apply(parse_model_type)
    df["category"] = df["experiment"].apply(parse_category)
    return df
