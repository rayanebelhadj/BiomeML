#!/usr/bin/env python3
"""
IBD Analysis - Experiment Runner
Runs multiple experiments with different configurations from experiments.yaml
"""

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
    """Load YAML configuration file"""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, filepath: Path):
    """Save configuration to YAML file"""
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """Deep merge two configuration dictionaries"""
    result = copy.deepcopy(base_config)
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def expand_grid_search(grid_config: Dict) -> List[Dict]:
    """Expand grid search configuration into list of individual configs"""
    parameters = grid_config.get('parameters', {})
    fixed = grid_config.get('fixed', {})
    
    # Get all parameter combinations
    param_names = list(parameters.keys())
    param_values = [parameters[name] for name in param_names]
    
    configs = []
    for values in itertools.product(*param_values):
        config = copy.deepcopy(fixed)
        
        # Add parameter values
        for name, value in zip(param_names, values):
            # Navigate nested structure
            if name in ['hidden_dim', 'num_layers', 'dropout']:
                if 'model_training' not in config:
                    config['model_training'] = {}
                if 'architecture' not in config['model_training']:
                    config['model_training']['architecture'] = {}
                config['model_training']['architecture'][name] = value
            elif name in ['batch_size', 'learning_rate', 'weight_decay']:
                if 'model_training' not in config:
                    config['model_training'] = {}
                if 'training' not in config['model_training']:
                    config['model_training']['training'] = {}
                config['model_training']['training'][name] = value
            elif name in ['knn_k', 'max_distance_factor']:
                if 'graph_construction' not in config:
                    config['graph_construction'] = {}
                if 'knn' not in config['graph_construction']:
                    config['graph_construction']['knn'] = {}
                # Map parameter names
                if name == 'knn_k':
                    config['graph_construction']['knn']['k'] = value
                elif name == 'max_distance_factor':
                    config['graph_construction']['knn']['max_distance_factor'] = value
        
        configs.append(config)
    
    return configs


def run_experiment(exp_name: str, config: Dict, base_config: Dict, 
                   notebooks_dir: Path, output_dir: Path, pixi_path: str = None,
                   num_runs: int = 1, base_seed: int = 42) -> Dict:
    """Run a single experiment with given configuration, optionally multiple times.
    
    Parameters:
    -----------
    exp_name : str
        Name of the experiment
    config : Dict
        Experiment-specific configuration overrides
    base_config : Dict
        Base configuration
    notebooks_dir : Path
        Directory containing notebooks
    output_dir : Path
        Output directory for results
    pixi_path : str
        Path to pixi executable
    num_runs : int
        Number of runs for this experiment (for confidence intervals)
    base_seed : int
        Base random seed (incremented for each run)
        
    Returns:
    --------
    Dict with experiment results (aggregated if num_runs > 1)
    """
    
    print(f"\n{'='*80}")
    print(f"🧪 Running Experiment: {exp_name}")
    if num_runs > 1:
        print(f"   Multiple runs: {num_runs}")
    print(f"{'='*80}")
    
    # Merge with base config
    full_config = merge_configs(base_config, config)
    
    # Handle multiple runs
    if num_runs > 1:
        return run_multiple_experiments(
            exp_name, config, base_config, full_config,
            notebooks_dir, output_dir, pixi_path,
            num_runs, base_seed
        )
    
    # Create experiment-specific output directory
    exp_output_dir = output_dir / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment config
    config_path = exp_output_dir / "config.yaml"
    save_yaml(full_config, config_path)
    print(f"📝 Saved config to: {config_path}")
    
    # Update config to use experiment-specific output
    full_config['data_extraction']['output']['base_dir'] = str(notebooks_dir / "IBD_analysis_output")
    
    # Save temporary config for notebooks to read
    temp_config_path = Path("config_temp.yaml")
    save_yaml(full_config, temp_config_path)
    
    # Determine command prefix (pixi or direct)
    if pixi_path:
        cmd_prefix = [pixi_path, "run"]
    else:
        cmd_prefix = []
    
    # Check if data extraction can be skipped using smart caching
    skip_data_extraction = False
    data_extraction_output = notebooks_dir / "IBD_analysis_output"
    
    # Check if data already extracted (simple file check)
    if (data_extraction_output / "biom_tables" / "AGP_IBD_cases.tsv").exists() and \
       (data_extraction_output / "phylogeny" / "MATRICES_IBD.pickle").exists():
        print(f"\n✅ Data already extracted - skipping step 1")
        skip_data_extraction = True
    else:
        print(f"\n🔄 Data not found - will run extraction (~48 minutes)")
        skip_data_extraction = False
    
    # Run notebooks in sequence
    notebooks = [
        "01_data_extraction.ipynb",
        "02_graph_construction.ipynb", 
        "03_model_training.ipynb"
    ]
    
    results = {
        'experiment': exp_name,
        'config': config,
        'start_time': datetime.now().isoformat(),
        'notebooks': {}
    }
    
    for notebook in notebooks:
        # Skip data extraction if already done
        if notebook == "01_data_extraction.ipynb" and skip_data_extraction:
            print(f"\n📓 Skipping: {notebook} (data already extracted)")
            results['notebooks'][notebook] = {
                'status': 'skipped',
                'reason': 'Data already extracted',
                'duration_seconds': 0
            }
            continue
        notebook_path = notebooks_dir / notebook
        output_notebook = exp_output_dir / f"{notebook.replace('.ipynb', '_executed.ipynb')}"
        
        print(f"\n📓 Executing: {notebook}")
        
        try:
            # Build nbconvert command
            if pixi_path:
                # Use pixi run python -m jupyter for proper environment
                cmd = [
                    pixi_path, "run", "python", "-m", "jupyter", "nbconvert",
                    "--to", "notebook",
                    "--execute",
                    str(notebook_path),
                    "--output", str(output_notebook),
                    "--ExecutePreprocessor.timeout=7200"  # 2 hour timeout
                ]
            else:
                # Direct jupyter command
                cmd = [
                    "jupyter", "nbconvert",
                    "--to", "notebook",
                    "--execute",
                    str(notebook_path),
                    "--output", str(output_notebook),
                    "--ExecutePreprocessor.timeout=7200"  # 2 hour timeout
                ]
            
            # Run notebook with config path in environment variable
            start = datetime.now()
            env = os.environ.copy()
            env['EXPERIMENT_CONFIG_PATH'] = str(exp_output_dir / "config.yaml")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env  # Pass environment with config path
            )
            duration = (datetime.now() - start).total_seconds()
            
            print(f"   ✅ Completed in {duration:.1f}s")
            
            results['notebooks'][notebook] = {
                'status': 'success',
                'duration_seconds': duration,
                'output_path': str(output_notebook)
            }
            
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed: {e}")
            print(f"   Error output: {e.stderr[:500]}")
            
            results['notebooks'][notebook] = {
                'status': 'failed',
                'error': str(e),
                'stderr': e.stderr[:1000]
            }
            
            # Stop pipeline on failure
            break
    
    results['end_time'] = datetime.now().isoformat()
    results['total_duration'] = (
        datetime.fromisoformat(results['end_time']) - 
        datetime.fromisoformat(results['start_time'])
    ).total_seconds()
    
    # Save results
    results_path = exp_output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Try to extract final metrics from model training output
    try:
        # Notebook saves to results/ relative to its execution directory
        # When executed from project root, this becomes notebooks/results/
        eval_results_path = notebooks_dir / "results" / "evaluation_results.json"
        
        if eval_results_path.exists():
            with open(eval_results_path, 'r') as f:
                eval_results = json.load(f)
            
            # Copy to experiment directory
            exp_eval_path = exp_output_dir / "evaluation_results.json"
            with open(exp_eval_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            # Clean up the temporary results file
            eval_results_path.unlink()
            
            results['metrics'] = eval_results
            print(f"\n📊 Final Metrics:")
            
            # Format metrics safely (handle missing values)
            test_acc = eval_results.get('test_accuracy', None)
            test_auc = eval_results.get('test_auc_roc', None)
            
            if test_acc is not None and isinstance(test_acc, (int, float)):
                print(f"   Test Accuracy: {test_acc:.4f}")
            else:
                print(f"   Test Accuracy: {test_acc if test_acc is not None else 'N/A'}")
            
            if test_auc is not None and isinstance(test_auc, (int, float)):
                print(f"   Test AUC-ROC: {test_auc:.4f}")
            else:
                print(f"   Test AUC-ROC: {test_auc if test_auc is not None else 'N/A'}")
    except Exception as e:
        print(f"   ⚠️ Could not extract metrics: {e}")
    
    # Clean up temp config
    if temp_config_path.exists():
        temp_config_path.unlink()
    
    return results


def run_multiple_experiments(
    exp_name: str, config: Dict, base_config: Dict, full_config: Dict,
    notebooks_dir: Path, output_dir: Path, pixi_path: str,
    num_runs: int, base_seed: int
) -> Dict:
    """Run an experiment multiple times with different seeds and aggregate results.
    
    Parameters:
    -----------
    exp_name : str
        Name of the experiment
    config : Dict
        Experiment-specific configuration
    base_config : Dict
        Base configuration
    full_config : Dict
        Merged configuration
    notebooks_dir : Path
        Directory containing notebooks
    output_dir : Path
        Output directory
    pixi_path : str
        Path to pixi executable
    num_runs : int
        Number of runs
    base_seed : int
        Base random seed
        
    Returns:
    --------
    Dict with aggregated results across all runs
    """
    import numpy as np
    
    all_run_results = []
    run_metrics = []
    
    for run_idx in range(num_runs):
        run_seed = base_seed + run_idx
        print(f"\n   📌 Run {run_idx + 1}/{num_runs} (seed={run_seed})")
        
        # Create run-specific config with updated seed
        run_config = copy.deepcopy(config)
        
        # Update seeds in config
        if 'data_extraction' not in run_config:
            run_config['data_extraction'] = {}
        if 'matching' not in run_config['data_extraction']:
            run_config['data_extraction']['matching'] = {}
        run_config['data_extraction']['matching']['random_seed'] = run_seed
        
        if 'model_training' not in run_config:
            run_config['model_training'] = {}
        if 'training' not in run_config['model_training']:
            run_config['model_training']['training'] = {}
        run_config['model_training']['training']['random_seed'] = run_seed
        
        # Create run-specific output directory
        run_output_dir = output_dir / exp_name / f"run_{run_idx}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run single experiment for this run
        run_full_config = merge_configs(base_config, run_config)
        
        # Save run config
        config_path = run_output_dir / "config.yaml"
        save_yaml(run_full_config, config_path)
        
        # Update config to use experiment-specific output
        run_full_config['data_extraction']['output']['base_dir'] = str(notebooks_dir / "IBD_analysis_output")
        
        # Save temporary config for notebooks
        temp_config_path = Path("config_temp.yaml")
        save_yaml(run_full_config, temp_config_path)
        
        # Determine command prefix
        if pixi_path:
            cmd_prefix = [pixi_path, "run"]
        else:
            cmd_prefix = []
        
        # Skip data extraction for runs after the first (reuse data)
        skip_data_extraction = run_idx > 0
        if not skip_data_extraction:
            data_extraction_output = notebooks_dir / "IBD_analysis_output"
            if (data_extraction_output / "biom_tables" / "AGP_IBD_cases.tsv").exists():
                skip_data_extraction = True
        
        # Run notebooks
        notebooks = [
            "01_data_extraction.ipynb",
            "02_graph_construction.ipynb", 
            "03_model_training.ipynb"
        ]
        
        run_result = {
            'run_idx': run_idx,
            'seed': run_seed,
            'start_time': datetime.now().isoformat(),
            'notebooks': {}
        }
        
        for notebook in notebooks:
            if notebook == "01_data_extraction.ipynb" and skip_data_extraction:
                run_result['notebooks'][notebook] = {
                    'status': 'skipped',
                    'reason': 'Data already extracted'
                }
                continue
            
            notebook_path = notebooks_dir / notebook
            output_notebook = run_output_dir / f"{notebook.replace('.ipynb', '_executed.ipynb')}"
            
            try:
                if pixi_path:
                    cmd = [
                        pixi_path, "run", "python", "-m", "jupyter", "nbconvert",
                        "--to", "notebook", "--execute",
                        str(notebook_path),
                        "--output", str(output_notebook),
                        "--ExecutePreprocessor.timeout=14400"
                    ]
                else:
                    cmd = [
                        "jupyter", "nbconvert",
                        "--to", "notebook", "--execute",
                        str(notebook_path),
                        "--output", str(output_notebook),
                        "--ExecutePreprocessor.timeout=14400"
                    ]
                
                start = datetime.now()
                env = os.environ.copy()
                env['EXPERIMENT_CONFIG_PATH'] = str(config_path)
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
                duration = (datetime.now() - start).total_seconds()
                
                run_result['notebooks'][notebook] = {
                    'status': 'success',
                    'duration_seconds': duration
                }
                
            except subprocess.CalledProcessError as e:
                run_result['notebooks'][notebook] = {
                    'status': 'failed',
                    'error': str(e)
                }
                break
        
        run_result['end_time'] = datetime.now().isoformat()
        
        # Extract metrics
        try:
            eval_path = notebooks_dir / "results" / "evaluation_results.json"
            if eval_path.exists():
                with open(eval_path, 'r') as f:
                    eval_results = json.load(f)
                run_result['metrics'] = eval_results
                run_metrics.append({
                    'test_accuracy': eval_results.get('test_accuracy'),
                    'test_auc': eval_results.get('test_auc_roc'),
                    'run_idx': run_idx
                })
                # Copy to run directory
                with open(run_output_dir / "evaluation_results.json", 'w') as f:
                    json.dump(eval_results, f, indent=2)
                eval_path.unlink()
        except Exception as e:
            print(f"      ⚠️ Could not extract metrics: {e}")
        
        # Save run result
        with open(run_output_dir / "results.json", 'w') as f:
            json.dump(run_result, f, indent=2)
        
        all_run_results.append(run_result)
        
        # Clean up temp config
        if temp_config_path.exists():
            temp_config_path.unlink()
    
    # Aggregate results across runs
    aggregated = aggregate_multi_run_results(all_run_results, run_metrics)
    aggregated['experiment'] = exp_name
    aggregated['config'] = config
    aggregated['num_runs'] = num_runs
    aggregated['base_seed'] = base_seed
    aggregated['per_run_results'] = all_run_results
    
    # Save aggregated results
    exp_output_dir = output_dir / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(exp_output_dir / "aggregated_results.json", 'w') as f:
        # Convert numpy types for JSON serialization
        json.dump(aggregated, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    
    # Also save as results.json for compatibility
    with open(exp_output_dir / "results.json", 'w') as f:
        json.dump(aggregated, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))
    
    # Print summary
    print(f"\n   📊 Aggregated Results ({num_runs} runs):")
    if 'metrics' in aggregated:
        metrics = aggregated['metrics']
        if 'test_accuracy' in metrics:
            acc = metrics['test_accuracy']
            print(f"      Accuracy: {acc['mean']:.4f} ± {acc['std']:.4f}")
        if 'test_auc' in metrics:
            auc = metrics['test_auc']
            if auc['mean'] is not None:
                print(f"      AUC: {auc['mean']:.4f} ± {auc['std']:.4f}")
    
    return aggregated


def aggregate_multi_run_results(all_run_results: List[Dict], run_metrics: List[Dict]) -> Dict:
    """Aggregate metrics across multiple runs.
    
    Parameters:
    -----------
    all_run_results : List[Dict]
        Results from each run
    run_metrics : List[Dict]
        Extracted metrics from each run
        
    Returns:
    --------
    Dict with aggregated statistics
    """
    import numpy as np
    from scipy import stats
    
    aggregated = {
        'start_time': all_run_results[0]['start_time'] if all_run_results else None,
        'end_time': all_run_results[-1]['end_time'] if all_run_results else None,
        'metrics': {}
    }
    
    if not run_metrics:
        return aggregated
    
    # Aggregate each metric
    for metric_name in ['test_accuracy', 'test_auc']:
        values = [m[metric_name] for m in run_metrics if m.get(metric_name) is not None]
        
        if values:
            values = np.array(values)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            
            # 95% confidence interval
            if len(values) > 1:
                ci = stats.t.interval(0.95, len(values) - 1, loc=mean, scale=stats.sem(values))
                ci_lower, ci_upper = float(ci[0]), float(ci[1])
            else:
                ci_lower, ci_upper = mean, mean
            
            aggregated['metrics'][metric_name] = {
                'mean': mean,
                'std': std,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'values': values.tolist(),
                'n': len(values)
            }
    
    return aggregated


def create_comparison_table(all_results: List[Dict], output_path: Path):
    """Create comparison table from all experiment results"""
    
    print(f"\n{'='*80}")
    print("📊 Creating Comparison Table")
    print(f"{'='*80}")
    
    rows = []
    for result in all_results:
        exp_name = result['experiment']
        
        # Extract metrics
        metrics = result.get('metrics', {})
        
        # Extract config parameters
        config = result.get('config', {})
        model_config = config.get('model_training', {})
        arch_config = model_config.get('architecture', {})
        train_config = model_config.get('training', {})
        
        row = {
            'experiment': exp_name,
            'test_accuracy': metrics.get('test_accuracy', None),
            'test_auc_roc': metrics.get('test_auc_roc', None),
            'val_accuracy': metrics.get('best_val_accuracy', None),
            'val_loss': metrics.get('best_val_loss', None),
            'hidden_dim': arch_config.get('hidden_dim', None),
            'num_layers': arch_config.get('num_layers', None),
            'dropout': arch_config.get('dropout', None),
            'batch_size': train_config.get('batch_size', None),
            'learning_rate': train_config.get('learning_rate', None),
            'epochs_trained': metrics.get('epochs_trained', None),
            'duration_seconds': result.get('total_duration', None),
            'status': 'success' if metrics else 'failed'
        }
        rows.append(row)
    
    # Create DataFrame and save
    import pandas as pd
    df = pd.DataFrame(rows)
    
    # Sort by test accuracy (descending)
    if 'test_accuracy' in df.columns:
        df = df.sort_values('test_accuracy', ascending=False)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Saved comparison table to: {output_path}")
    
    # Print summary
    print(f"\n📈 Experiment Summary:")
    print(df.to_string(index=False))
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Run IBD analysis experiments')
    parser.add_argument('--list', action='store_true',
                       help='List all available experiments')
    parser.add_argument('--all', action='store_true',
                       help='Run ALL experiments (166 total)')
    parser.add_argument('--experiments', nargs='+', 
                       help='Specific experiments to run')
    parser.add_argument('--grid', action='store_true',
                       help='Run grid search experiments')
    parser.add_argument('--output-dir', type=str, default='experiments',
                       help='Output directory for results')
    parser.add_argument('--pixi', type=str, default=None,
                       help='Path to pixi executable (e.g., ~/.pixi/bin/pixi)')
    parser.add_argument('--notebooks-dir', type=str, default='notebooks',
                       help='Directory containing notebooks')
    parser.add_argument('--num-runs', type=int, default=1,
                       help='Number of runs per experiment (for confidence intervals)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed (incremented for each run)')
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent  # Go up from scripts/ to project root
    config_path = project_root / "config.yaml"
    experiments_path = project_root / "experiments.yaml"
    notebooks_dir = project_root / args.notebooks_dir
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configurations
    print("Loading configurations...")
    base_config = load_yaml(config_path)
    experiments_config = load_yaml(experiments_path)
    
    # Filter out non-experiment entries (like grid searches)
    all_experiments = {k: v for k, v in experiments_config.items() 
                       if isinstance(v, dict) and not k.startswith('grid_')}
    
    # Handle --list: show available experiments
    if args.list:
        print(f"\nAvailable experiments ({len(all_experiments)} total):\n")
        for name in sorted(all_experiments.keys()):
            desc = all_experiments[name].get('description', '')
            print(f"  {name:<30} {desc[:50]}")
        print(f"\nUsage:")
        print(f"  pixi run python scripts/run_experiments.py --experiments baseline")
        print(f"  pixi run python scripts/run_experiments.py --all --num-runs 50")
        return
    
    # Determine which experiments to run
    experiments_to_run = []
    
    if args.all:
        # Run ALL experiments
        for exp_name, exp_config in all_experiments.items():
            experiments_to_run.append((exp_name, exp_config))
    
    elif args.grid:
        # Run grid search experiments
        for grid_name in ['grid_architecture', 'grid_training', 'grid_graphs']:
            if grid_name in experiments_config:
                print(f"Expanding grid search: {grid_name}")
                grid_configs = expand_grid_search(experiments_config[grid_name])
                for i, config in enumerate(grid_configs):
                    exp_name = f"{grid_name}_{i+1:03d}"
                    experiments_to_run.append((exp_name, config))
                print(f"   Generated {len(grid_configs)} configurations")
    
    elif args.experiments:
        # Run specific experiments
        for exp_name in args.experiments:
            if exp_name in all_experiments:
                experiments_to_run.append((exp_name, all_experiments[exp_name]))
            else:
                print(f"Warning: Experiment '{exp_name}' not found in experiments.yaml")
    
    else:
        # No arguments: show usage
        print(f"\nNo experiments specified. Use one of:")
        print(f"  --list                    List all {len(all_experiments)} experiments")
        print(f"  --experiments <names>     Run specific experiments")
        print(f"  --all                     Run ALL experiments")
        print(f"\nExample:")
        print(f"  pixi run python scripts/run_experiments.py --list")
        print(f"  pixi run python scripts/run_experiments.py --experiments baseline ibd_gcn")
        print(f"  pixi run python scripts/run_experiments.py --all --num-runs 50")
        return
    
    if not experiments_to_run:
        print("No experiments to run!")
        return
    
    print(f"\nRunning {len(experiments_to_run)} experiments")
    print(f"Output directory: {output_dir}")
    
    # Run all experiments
    all_results = []
    for exp_name, exp_config in experiments_to_run:
        try:
            result = run_experiment(
                exp_name, 
                exp_config, 
                base_config, 
                notebooks_dir, 
                output_dir,
                pixi_path=args.pixi,
                num_runs=args.num_runs,
                base_seed=args.seed
            )
            all_results.append(result)
        except Exception as e:
            print(f"❌ Experiment {exp_name} failed with error: {e}")
            all_results.append({
                'experiment': exp_name,
                'status': 'failed',
                'error': str(e)
            })
    
    # Create comparison table
    comparison_path = output_dir / "comparison_table.csv"
    create_comparison_table(all_results, comparison_path)
    
    # Save all results
    all_results_path = output_dir / "all_results.json"
    with open(all_results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ All experiments complete!")
    print(f"{'='*80}")
    print(f"📊 Results saved to: {output_dir}")
    print(f"📈 Comparison table: {comparison_path}")
    print(f"📋 Full results: {all_results_path}")


if __name__ == "__main__":
    main()

