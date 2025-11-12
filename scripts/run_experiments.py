#!/usr/bin/env python3
"""Run experiments from experiments.yaml"""

import yaml
import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import copy
import itertools
import argparse


def load_yaml(filepath: Path) -> Dict:
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, filepath: Path):
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    result = copy.deepcopy(base_config)
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def expand_grid_search(grid_config: Dict) -> List[Dict]:
    parameters = grid_config.get('parameters', {})
    fixed = grid_config.get('fixed', {})
    param_names = list(parameters.keys())
    param_values = [parameters[name] for name in param_names]
    configs = []
    for values in itertools.product(*param_values):
        config = copy.deepcopy(fixed)
        for name, value in zip(param_names, values):
            if name in ['hidden_dim', 'num_layers', 'dropout']:
                if 'model_training' not in config: config['model_training'] = {}
                if 'architecture' not in config['model_training']: config['model_training']['architecture'] = {}
                config['model_training']['architecture'][name] = value
            elif name in ['batch_size', 'learning_rate', 'weight_decay']:
                if 'model_training' not in config: config['model_training'] = {}
                if 'training' not in config['model_training']: config['model_training']['training'] = {}
                config['model_training']['training'][name] = value
            elif name in ['knn_k', 'max_distance_factor']:
                if 'graph_construction' not in config: config['graph_construction'] = {}
                if 'knn' not in config['graph_construction']: config['graph_construction']['knn'] = {}
                if name == 'knn_k': config['graph_construction']['knn']['k'] = value
                elif name == 'max_distance_factor': config['graph_construction']['knn']['max_distance_factor'] = value
        configs.append(config)
    return configs


def run_experiment(exp_name: str, config: Dict, base_config: Dict,
                   notebooks_dir: Path, output_dir: Path, pixi_path: str = None,
                   num_runs: int = 1, base_seed: int = 42, parallel_workers: int = 4) -> Dict:
    print(f"\n{'='*80}")
    print(f"Running Experiment: {exp_name}")
    print(f"{'='*80}")

    full_config = merge_configs(base_config, config)

    exp_output_dir = output_dir / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    config_path = exp_output_dir / "config.yaml"
    save_yaml(full_config, config_path)

    disease = full_config.get('disease') or full_config.get('data_extraction', {}).get('disease', 'IBD')
    disease = disease.upper()

    if "disease_criteria" not in full_config["data_extraction"]:
        full_config["data_extraction"]["disease_criteria"] = {}
    full_config["data_extraction"]["disease_criteria"]["disease"] = disease

    full_config['data_extraction']['output']['base_dir'] = str(notebooks_dir / f"{disease}_analysis_output")
    full_config['output_dir'] = str(exp_output_dir)
    save_yaml(full_config, config_path)

    temp_config_path = Path("config_temp.yaml")
    save_yaml(full_config, temp_config_path)

    if pixi_path:
        cmd_prefix = [pixi_path, "run"]
    else:
        cmd_prefix = []

    data_extraction_output = notebooks_dir / f"{disease}_analysis_output"
    skip_data_extraction = (
        (data_extraction_output / "biom_tables" / f"AGP_{disease}_cases.tsv").exists() and
        (data_extraction_output / "phylogeny" / f"MATRICES_{disease}.pickle").exists()
    )

    notebooks = [
        "01_data_extraction.ipynb",
        "02_graph_construction.ipynb",
        "03_model_training.ipynb"
    ]

    results = {
        'experiment': exp_name, 'config': config,
        'start_time': datetime.now().isoformat(), 'notebooks': {}
    }

    for notebook in notebooks:
        if notebook == "01_data_extraction.ipynb" and skip_data_extraction:
            results['notebooks'][notebook] = {'status': 'skipped', 'reason': 'Data already extracted', 'duration_seconds': 0}
            continue
        notebook_path = notebooks_dir / notebook
        output_notebook = exp_output_dir / f"{notebook.replace('.ipynb', '_executed.ipynb')}"
        print(f"\nExecuting: {notebook}")
        try:
            if pixi_path:
                cmd = [pixi_path, "run", "python", "-m", "nbconvert", "--to", "notebook",
                       "--execute", str(notebook_path), "--output", str(output_notebook),
                       "--ExecutePreprocessor.timeout=7200"]
            else:
                cmd = ["python", "-m", "nbconvert", "--to", "notebook", "--execute",
                       str(notebook_path), "--output", str(output_notebook),
                       "--ExecutePreprocessor.timeout=7200"]
            start = datetime.now()
            env = os.environ.copy()
            env['EXPERIMENT_CONFIG_PATH'] = str(config_path)
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            duration = (datetime.now() - start).total_seconds()
            results['notebooks'][notebook] = {'status': 'success', 'duration_seconds': duration}
        except subprocess.CalledProcessError as e:
            results['notebooks'][notebook] = {'status': 'failed', 'error': str(e), 'stderr': e.stderr[:1000]}
            break

    results['end_time'] = datetime.now().isoformat()
    results_path = exp_output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    if temp_config_path.exists():
        temp_config_path.unlink()

    return results


def create_comparison_table(all_results: List[Dict], output_path: Path):
    import pandas as pd
    rows = []
    for result in all_results:
        exp_name = result['experiment']
        metrics = result.get('metrics', {})
        config = result.get('config', {})
        model_config = config.get('model_training', {})
        arch_config = model_config.get('architecture', {})
        train_config = model_config.get('training', {})
        row = {
            'experiment': exp_name,
            'test_accuracy': metrics.get('test_accuracy', None),
            'test_auc_roc': metrics.get('test_auc_roc', None),
            'hidden_dim': arch_config.get('hidden_dim', None),
            'num_layers': arch_config.get('num_layers', None),
            'status': 'success' if metrics else 'failed'
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if 'test_accuracy' in df.columns:
        # BUG: crashes if test_accuracy is a dict (multi-run results)
        df = df.sort_values('test_accuracy', ascending=False)
    df.to_csv(output_path, index=False)
    print(f"Saved comparison table to: {output_path}")
    return df


def main():
    parser = argparse.ArgumentParser(description='Run BiomeML experiments')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--experiments', nargs='+')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--output-dir', type=str, default='experiments')
    parser.add_argument('--pixi', type=str, default=None)
    parser.add_argument('--notebooks-dir', type=str, default='notebooks')
    parser.add_argument('--num-runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.yaml"
    experiments_path = project_root / "experiments.yaml"
    notebooks_dir = project_root / args.notebooks_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_config = load_yaml(config_path)
    experiments_config = load_yaml(experiments_path)
    all_experiments = {k: v for k, v in experiments_config.items()
                       if isinstance(v, dict) and not k.startswith('grid_')}

    if args.list:
        print(f"\nAvailable experiments ({len(all_experiments)} total):")
        for name in sorted(all_experiments.keys()):
            desc = all_experiments[name].get('description', '')
            print(f"  {name:<30} {desc[:50]}")
        return

    experiments_to_run = []
    if args.all:
        for exp_name, exp_config in all_experiments.items():
            experiments_to_run.append((exp_name, exp_config))
    elif args.grid:
        for grid_name in ['grid_architecture', 'grid_training', 'grid_graphs']:
            if grid_name in experiments_config:
                grid_configs = expand_grid_search(experiments_config[grid_name])
                for i, config in enumerate(grid_configs):
                    experiments_to_run.append((f"{grid_name}_{i+1:03d}", config))
    elif args.experiments:
        for exp_name in args.experiments:
            if exp_name in all_experiments:
                experiments_to_run.append((exp_name, all_experiments[exp_name]))
    else:
        print("No experiments specified. Use --list, --experiments, or --all")
        return

    all_results = []
    for exp_name, exp_config in experiments_to_run:
        try:
            result = run_experiment(
                exp_name, exp_config, base_config, notebooks_dir, output_dir,
                pixi_path=args.pixi, num_runs=args.num_runs, base_seed=args.seed
            )
            all_results.append(result)
        except Exception as e:
            all_results.append({'experiment': exp_name, 'status': 'failed', 'error': str(e)})

    comparison_path = output_dir / "comparison_table.csv"
    create_comparison_table(all_results, comparison_path)

    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll experiments complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
