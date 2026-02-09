#!/usr/bin/env python3
"""
Cross-validation module for BiomeML experiments.
Provides k-fold cross-validation with proper stratification.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from src.models import get_loss, WeightedBCELoss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from scipy.stats import ttest_rel
from pathlib import Path
import json
from typing import Dict, List, Any, Callable, Optional
from datetime import datetime
import copy


def create_dataset_from_graphs(graphs, labels, dataset_class, **dataset_kwargs):
    """Create PyTorch Geometric dataset from graphs and labels."""
    return dataset_class(graphs, labels, **dataset_kwargs)


def train_epoch(model, loader, optimizer, criterion, device, num_classes=2):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch)
        
        if num_classes == 2:
            loss = criterion(out, batch.y)
        else:
            loss = criterion(out, batch.y.long())
        
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"NaN/Inf loss detected during training (loss={loss.item()}). "
                f"Check learning rate, input data, or model architecture."
            )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
        if num_classes == 2:
            pred = (out > 0.5).float()
            correct += (pred == batch.y).sum().item()
        else:
            pred = out.argmax(dim=1)
            correct += (pred == batch.y.long()).sum().item()
        total += batch.num_graphs
    
    return (total_loss / total if total > 0 else 0.0, 
            correct / total if total > 0 else 0.0)


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=2):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in loader:
        batch = batch.to(device)
        out = model(batch)
        
        if num_classes == 2:
            loss = criterion(out, batch.y)
        else:
            loss = criterion(out, batch.y.long())
        
        total_loss += loss.item() * batch.num_graphs
        
        if num_classes == 2:
            pred = (out > 0.5).float()
            correct += (pred == batch.y).sum().item()
            all_probs.extend(out.cpu().numpy())
        else:
            pred = out.argmax(dim=1)
            correct += (pred == batch.y.long()).sum().item()
            all_probs.extend(torch.softmax(out, dim=1).cpu().numpy())
        
        total += batch.num_graphs
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Add balanced accuracy
    balanced_acc = balanced_accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    
    return (
        total_loss / total if total > 0 else 0.0,
        accuracy,
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
        balanced_acc  # NEW: balanced accuracy
    )


def train_single_fold(
    train_graphs, train_labels,
    val_graphs, val_labels,
    model_class, model_params,
    training_params, dataset_class,
    device, num_classes=2,
    dataset_kwargs=None
):
    """Train model for a single fold."""
    if dataset_kwargs is None:
        dataset_kwargs = {}
    
    # Create datasets
    train_dataset = create_dataset_from_graphs(
        train_graphs, train_labels, dataset_class, **dataset_kwargs
    )
    val_dataset = create_dataset_from_graphs(
        val_graphs, val_labels, dataset_class, **dataset_kwargs
    )
    
    # Create loaders
    batch_size = training_params['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    model = model_class(**model_params).to(device)
    
    # Setup training
    lr = training_params['learning_rate']
    weight_decay = training_params['weight_decay']
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    loss_cfg = training_params['loss']
    if isinstance(loss_cfg, dict):
        loss_type = loss_cfg['type']
    else:
        raise ValueError(
            f"training_params['loss'] must be a dict with at least a 'type' key, got: {type(loss_cfg)}"
        )
    loss_kwargs = {k: v for k, v in loss_cfg.items() if k != 'type'}
    if 'focal_gamma' in loss_kwargs:
        loss_kwargs['gamma'] = loss_kwargs.pop('focal_gamma')

    if num_classes == 2:
        if loss_type == 'weighted_bce':
            loss_kwargs['pos_weight'] = WeightedBCELoss.compute_pos_weight(train_labels)
        criterion = get_loss(loss_type, **loss_kwargs) if loss_type in ('bce', 'focal', 'weighted_bce') else get_loss('bce')
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Training loop
    num_epochs = training_params['num_epochs']
    patience = training_params['early_stopping_patience']
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, num_classes
        )
        val_loss, val_acc, _, _, _, _ = evaluate(
            model, val_loader, criterion, device, num_classes
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    val_loss, val_acc, val_preds, val_labels_out, val_probs, val_balanced_acc = evaluate(
        model, val_loader, criterion, device, num_classes
    )
    
    # Calculate AUC
    try:
        if num_classes == 2:
            val_auc = roc_auc_score(val_labels_out, val_probs)
        else:
            val_auc = roc_auc_score(val_labels_out, val_probs, multi_class='ovr')
    except ValueError:
        val_auc = None
    
    return {
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'val_balanced_accuracy': val_balanced_acc,  # NEW
        'val_auc': val_auc,
        'model_state': best_model_state,
        'predictions': val_preds.tolist(),
        'labels': val_labels_out.tolist(),
        'probabilities': val_probs.tolist() if isinstance(val_probs, np.ndarray) else val_probs
    }


def run_kfold_experiment(
    graphs: List,
    labels: np.ndarray,
    model_class,
    model_params: Dict,
    training_params: Dict,
    n_folds: int = 5,
    num_runs_per_fold: int = 1,
    random_seed: int = 42,
    device: str = 'cuda',
    num_classes: int = 2,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
    dataset_class = None,
    dataset_kwargs: Dict = None
) -> Dict:
    """
    Run k-fold cross-validation experiment.
    
    Parameters
    ----------
    graphs : List
        List of NetworkX graphs
    labels : np.ndarray
        Array of labels
    model_class : class
        Model class to instantiate
    model_params : Dict
        Model initialization parameters
    training_params : Dict
        Training parameters (batch_size, learning_rate, etc.)
    n_folds : int
        Number of folds
    num_runs_per_fold : int
        Number of runs per fold (for uncertainty estimation)
    random_seed : int
        Random seed for reproducibility
    device : str
        Device to use ('cuda' or 'cpu')
    num_classes : int
        Number of classes
    output_dir : Path, optional
        Output directory for saving results
    verbose : bool
        Whether to print progress
    dataset_class : class
        Dataset class to use (should have same interface as MicrobiomeGraphDataset)
    dataset_kwargs : Dict
        Additional kwargs for dataset creation
        
    Returns
    -------
    Dict
        Results including per-fold metrics and aggregated statistics
    """
    if dataset_kwargs is None:
        dataset_kwargs = {}
    
    if dataset_class is None:
        raise ValueError("dataset_class must be provided")
    if len(graphs) == 0:
        raise ValueError("graphs list is empty")
    if len(labels) == 0:
        raise ValueError("labels array is empty")
    if len(graphs) != len(labels):
        raise ValueError(
            f"graphs ({len(graphs)}) and labels ({len(labels)}) must have same length"
        )
    
    graphs = np.array(graphs, dtype=object)
    labels = np.array(labels)
    
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        raise ValueError(
            f"labels must contain at least 2 classes, got {len(unique_classes)}: {unique_classes}"
        )
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if n_folds > len(labels):
        raise ValueError(
            f"n_folds ({n_folds}) cannot exceed number of samples ({len(labels)})"
        )
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    fold_results = []
    all_predictions = []
    all_true_labels = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(graphs, labels)):
        if verbose:
            print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")
        
        train_graphs = graphs[train_idx].tolist()
        train_labels = labels[train_idx]
        val_graphs = graphs[val_idx].tolist()
        val_labels = labels[val_idx]
        
        fold_runs = []
        for run_idx in range(num_runs_per_fold):
            run_seed = random_seed + fold_idx * 100 + run_idx
            torch.manual_seed(run_seed)
            np.random.seed(run_seed)
            
            result = train_single_fold(
                train_graphs, train_labels,
                val_graphs, val_labels,
                model_class, model_params,
                training_params, dataset_class,
                device, num_classes,
                dataset_kwargs
            )
            fold_runs.append(result)
            
            if verbose:
                print(f"  Run {run_idx + 1}: Acc={result['val_accuracy']:.4f}, "
                      f"AUC={result['val_auc']:.4f if result['val_auc'] else 'N/A'}")
        
        # Aggregate runs within fold
        fold_accuracies = [r['val_accuracy'] for r in fold_runs]
        fold_balanced_accs = [r['val_balanced_accuracy'] for r in fold_runs]
        fold_aucs = [r['val_auc'] for r in fold_runs if r['val_auc'] is not None]
        
        fold_result = {
            'fold_idx': fold_idx,
            'mean_accuracy': np.mean(fold_accuracies),
            'std_accuracy': np.std(fold_accuracies) if len(fold_accuracies) > 1 else 0,
            'mean_balanced_accuracy': np.mean(fold_balanced_accs),
            'std_balanced_accuracy': np.std(fold_balanced_accs) if len(fold_balanced_accs) > 1 else 0,
            'mean_auc': np.mean(fold_aucs) if fold_aucs else None,
            'std_auc': np.std(fold_aucs) if len(fold_aucs) > 1 else 0,
            'runs': fold_runs,
            'train_size': len(train_idx),
            'val_size': len(val_idx)
        }
        fold_results.append(fold_result)
        
        # Collect predictions from best run
        best_run = max(fold_runs, key=lambda x: x['val_accuracy'])
        all_predictions.extend(best_run['predictions'])
        all_true_labels.extend(best_run['labels'])
        
        if verbose:
            print(f"  Fold {fold_idx + 1} Mean: Acc={fold_result['mean_accuracy']:.4f}")
    
    # Aggregate across folds
    all_accuracies = [f['mean_accuracy'] for f in fold_results]
    all_balanced_accs = [f['mean_balanced_accuracy'] for f in fold_results]
    all_aucs = [f['mean_auc'] for f in fold_results if f['mean_auc'] is not None]
    
    aggregated = {
        'mean_accuracy': np.mean(all_accuracies),
        'std_accuracy': np.std(all_accuracies),
        'mean_balanced_accuracy': np.mean(all_balanced_accs),
        'std_balanced_accuracy': np.std(all_balanced_accs),
        'mean_auc': np.mean(all_aucs) if all_aucs else None,
        'std_auc': np.std(all_aucs) if all_aucs else None,
        'n_folds': n_folds,
        'num_runs_per_fold': num_runs_per_fold
    }
    
    # Calculate 95% confidence interval
    if len(all_accuracies) > 1:
        from scipy import stats
        ci = stats.t.interval(
            0.95, len(all_accuracies) - 1,
            loc=aggregated['mean_accuracy'],
            scale=stats.sem(all_accuracies)
        )
        aggregated['accuracy_ci_lower'] = ci[0]
        aggregated['accuracy_ci_upper'] = ci[1]
    
    results = {
        'fold_results': fold_results,
        'aggregated': aggregated,
        'all_predictions': all_predictions,
        'all_true_labels': all_true_labels,
        'config': {
            'n_folds': n_folds,
            'num_runs_per_fold': num_runs_per_fold,
            'random_seed': random_seed,
            'num_classes': num_classes,
            'model_params': model_params,
            'training_params': training_params
        },
        'timestamp': datetime.now().isoformat()
    }
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"Cross-Validation Complete!")
        print(f"Mean Accuracy: {aggregated['mean_accuracy']:.4f} ± {aggregated['std_accuracy']:.4f}")
        if aggregated['mean_auc']:
            print(f"Mean AUC: {aggregated['mean_auc']:.4f} ± {aggregated['std_auc']:.4f}")
        print(f"{'='*50}")
    
    return results


def save_cv_results(results: Dict, output_path: Path):
    """Save cross-validation results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        else:
            return obj
    
    results_clean = convert_numpy(results)
    
    # Remove model states (not JSON serializable)
    for fold in results_clean.get('fold_results', []):
        for run in fold.get('runs', []):
            run.pop('model_state', None)
    
    with open(output_path, 'w') as f:
        json.dump(results_clean, f, indent=2)


# =============================================================================
# FDR CORRECTION FOR MULTIPLE COMPARISONS
# =============================================================================

def compare_experiments_with_fdr(
    results_dict: Dict[str, Dict],
    baseline_key: str = 'baseline',
    metric: str = 'accuracies'
) -> Dict:
    """
    Compare experiments using PAIRED t-test (same splits/seeds).
    
    IMPORTANT: Uses ttest_rel (paired) not ttest_ind (independent)
    because experiments share the same train/test splits.
    
    Parameters
    ----------
    results_dict : Dict[str, Dict]
        Dictionary mapping experiment names to results.
        Each result should have a list of accuracies (one per run).
        Example: {'baseline': {'accuracies': [0.6, 0.62, ...]}, ...}
    baseline_key : str
        Key of the baseline experiment to compare against
    metric : str
        Metric key to compare ('accuracies', 'balanced_accuracies', 'aucs')
    
    Returns
    -------
    Dict
        Comparison results with corrected p-values
    """
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError:
        print("WARNING: statsmodels not installed, skipping FDR correction")
        return {'error': 'statsmodels not installed'}
    
    if baseline_key not in results_dict:
        return {'error': f'Baseline key {baseline_key} not found'}
    
    baseline_values = results_dict[baseline_key].get(metric, [])
    if not baseline_values:
        return {'error': f'No {metric} found for baseline'}
    
    p_values = []
    comparisons = []
    effect_sizes = []
    
    for exp_name, exp_results in results_dict.items():
        if exp_name == baseline_key:
            continue
        
        exp_values = exp_results.get(metric, [])
        
        if len(exp_values) != len(baseline_values):
            print(f"WARNING: {exp_name} has {len(exp_values)} values, "
                  f"baseline has {len(baseline_values)}. Skipping.")
            continue
        
        # PAIRED t-test (same splits/seeds)
        _, p = ttest_rel(baseline_values, exp_values)
        p_values.append(p)
        comparisons.append(exp_name)
        
        # Effect size (Cohen's d for paired samples)
        diff = np.array(baseline_values) - np.array(exp_values)
        effect_size = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        effect_sizes.append(effect_size)
    
    if not p_values:
        return {'error': 'No valid comparisons found'}
    
    # Benjamini-Hochberg FDR correction
    rejected, corrected_p, _, _ = multipletests(p_values, method='fdr_bh')
    
    return {
        'comparisons': comparisons,
        'raw_p_values': [float(p) for p in p_values],
        'corrected_p_values': [float(p) for p in corrected_p],
        'significant': [bool(r) for r in rejected],
        'effect_sizes': [float(e) for e in effect_sizes],
        'test_type': 'paired_ttest (ttest_rel)',
        'correction_method': 'Benjamini-Hochberg FDR',
        'paired_by': '(seed, split_id)',  # Document alignment
        'n_comparisons': len(comparisons),
        'baseline': baseline_key,
        'metric': metric
    }


if __name__ == "__main__":
    print("Cross-validation module loaded successfully")
    print("Available functions: run_kfold_experiment, save_cv_results, compare_experiments_with_fdr")

