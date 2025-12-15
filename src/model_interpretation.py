#!/usr/bin/env python3
"""
Model interpretation module for BiomeML GNN experiments.
Provides gradient-based feature importance and visualization tools.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
import warnings


def compute_node_gradients(
    model,
    data: Data,
    target_class: int = 1,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Compute gradient-based node importance using backpropagation.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained GNN model
    data : Data
        PyTorch Geometric Data object
    target_class : int
        Target class for gradient computation
    device : str
        Device to use
        
    Returns
    -------
    np.ndarray
        Node importance scores (gradient magnitudes)
    """
    model.eval()
    data = data.to(device)
    
    # Enable gradients for node features
    data.x.requires_grad_(True)
    
    # Forward pass
    output = model(data)
    
    # For binary classification, use output directly
    if output.dim() == 0 or (output.dim() == 1 and output.size(0) == 1):
        if target_class == 1:
            loss = output.squeeze()
        else:
            loss = 1 - output.squeeze()
    else:
        # Multi-class: select target class
        loss = output[0, target_class] if output.dim() > 1 else output[target_class]
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Get gradients
    gradients = data.x.grad.detach().cpu().numpy()
    
    # Compute importance as gradient magnitude
    importance = np.abs(gradients).sum(axis=1)
    
    return importance


def compute_integrated_gradients(
    model,
    data: Data,
    target_class: int = 1,
    steps: int = 50,
    baseline: Optional[torch.Tensor] = None,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Compute Integrated Gradients for node importance.
    
    More robust than simple gradients - attributes importance by
    integrating gradients along path from baseline to input.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained GNN model
    data : Data
        PyTorch Geometric Data object
    target_class : int
        Target class
    steps : int
        Number of interpolation steps
    baseline : torch.Tensor, optional
        Baseline input (default: zeros)
    device : str
        Device to use
        
    Returns
    -------
    np.ndarray
        Node importance scores
    """
    model.eval()
    data = data.to(device)
    
    if baseline is None:
        baseline = torch.zeros_like(data.x)
    else:
        baseline = baseline.to(device)
    
    # Scale inputs along interpolation path
    scaled_inputs = []
    for step in range(steps + 1):
        alpha = step / steps
        scaled_input = baseline + alpha * (data.x - baseline)
        scaled_inputs.append(scaled_input)
    
    # Compute gradients at each step
    gradients = []
    for scaled_x in scaled_inputs:
        # Create copy of data with scaled features
        scaled_data = Data(
            x=scaled_x.clone().requires_grad_(True),
            edge_index=data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
            batch=data.batch if hasattr(data, 'batch') else torch.zeros(scaled_x.size(0), dtype=torch.long, device=device)
        )
        
        output = model(scaled_data)
        
        if output.dim() == 0 or (output.dim() == 1 and output.size(0) == 1):
            loss = output.squeeze() if target_class == 1 else 1 - output.squeeze()
        else:
            loss = output[0, target_class] if output.dim() > 1 else output[target_class]
        
        model.zero_grad()
        loss.backward()
        
        gradients.append(scaled_data.x.grad.detach().clone())
    
    # Average gradients
    avg_gradients = torch.stack(gradients).mean(dim=0)
    
    # Multiply by (input - baseline)
    integrated_gradients = (data.x - baseline) * avg_gradients
    
    # Sum over feature dimension for node importance
    importance = integrated_gradients.abs().sum(dim=1).cpu().numpy()
    
    return importance


def get_top_important_nodes(
    importance: np.ndarray,
    node_ids: Optional[List] = None,
    top_k: int = 20
) -> List[Tuple]:
    """
    Get top-k most important nodes.
    
    Parameters
    ----------
    importance : np.ndarray
        Node importance scores
    node_ids : List, optional
        List of node identifiers (e.g., phylogenetic IDs)
    top_k : int
        Number of top nodes to return
        
    Returns
    -------
    List[Tuple]
        List of (node_id/index, importance_score) tuples
    """
    top_indices = np.argsort(importance)[::-1][:top_k]
    
    if node_ids is not None:
        return [(node_ids[i], importance[i]) for i in top_indices]
    else:
        return [(i, importance[i]) for i in top_indices]


def aggregate_importance_across_samples(
    importance_list: List[np.ndarray],
    method: str = 'mean'
) -> np.ndarray:
    """
    Aggregate node importance across multiple samples.
    
    Parameters
    ----------
    importance_list : List[np.ndarray]
        List of importance arrays
    method : str
        Aggregation method ('mean', 'median', 'max')
        
    Returns
    -------
    np.ndarray
        Aggregated importance scores
    """
    # Pad arrays to same length
    max_len = max(len(imp) for imp in importance_list)
    padded = []
    for imp in importance_list:
        padded_imp = np.zeros(max_len)
        padded_imp[:len(imp)] = imp
        padded.append(padded_imp)
    
    stacked = np.stack(padded)
    
    if method == 'mean':
        return np.mean(stacked, axis=0)
    elif method == 'median':
        return np.median(stacked, axis=0)
    elif method == 'max':
        return np.max(stacked, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_prediction(
    model,
    data: Data,
    device: str = 'cuda',
    method: str = 'gradient'
) -> Dict:
    """
    Analyze a single prediction with importance scores.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained model
    data : Data
        Input data
    device : str
        Device
    method : str
        'gradient' or 'integrated_gradients'
        
    Returns
    -------
    Dict
        Analysis results including prediction, confidence, and importance
    """
    model.eval()
    data = data.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(data)
        if output.dim() == 0:
            prob = output.item()
        elif output.dim() == 1:
            prob = output[0].item() if output.size(0) == 1 else output.cpu().numpy()
        else:
            prob = F.softmax(output, dim=1).cpu().numpy()
    
    # Compute importance
    if method == 'gradient':
        importance = compute_node_gradients(model, data, device=device)
    elif method == 'integrated_gradients':
        importance = compute_integrated_gradients(model, data, device=device)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Determine prediction
    if isinstance(prob, float):
        predicted_class = 1 if prob > 0.5 else 0
        confidence = prob if predicted_class == 1 else 1 - prob
    else:
        predicted_class = np.argmax(prob)
        confidence = prob[predicted_class] if prob.ndim == 1 else prob[0, predicted_class]
    
    return {
        'predicted_class': int(predicted_class),
        'confidence': float(confidence),
        'probability': prob if isinstance(prob, float) else prob.tolist(),
        'node_importance': importance.tolist(),
        'top_nodes': get_top_important_nodes(importance, top_k=10),
        'method': method,
        'num_nodes': len(importance)
    }


def batch_analyze_predictions(
    model,
    dataset,
    device: str = 'cuda',
    method: str = 'gradient',
    max_samples: int = 100
) -> List[Dict]:
    """
    Analyze predictions for multiple samples.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained model
    dataset : Dataset
        PyTorch Geometric dataset
    device : str
        Device
    method : str
        Importance method
    max_samples : int
        Maximum samples to analyze
        
    Returns
    -------
    List[Dict]
        List of analysis results
    """
    results = []
    n_samples = min(len(dataset), max_samples)
    
    for i in range(n_samples):
        try:
            data = dataset[i]
            analysis = analyze_prediction(model, data, device, method)
            analysis['sample_idx'] = i
            analysis['true_label'] = int(data.y.item()) if hasattr(data, 'y') else None
            results.append(analysis)
        except Exception as e:
            warnings.warn(f"Error analyzing sample {i}: {e}")
    
    return results


def visualize_node_importance(
    importance: np.ndarray,
    node_ids: Optional[List] = None,
    top_k: int = 20,
    title: str = "Node Importance",
    ax=None
):
    """
    Visualize node importance as a bar chart.
    
    Parameters
    ----------
    importance : np.ndarray
        Node importance scores
    node_ids : List, optional
        Node identifiers
    top_k : int
        Number of top nodes to show
    title : str
        Plot title
    ax : matplotlib axes, optional
        Axes to plot on
        
    Returns
    -------
    matplotlib axes
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    top_nodes = get_top_important_nodes(importance, node_ids, top_k)
    
    labels = [str(n[0])[:20] for n in top_nodes]  # Truncate long labels
    scores = [n[1] for n in top_nodes]
    
    bars = ax.barh(range(len(labels)), scores, color='steelblue')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.invert_yaxis()  # Highest at top
    
    return ax


def visualize_embedding_space(
    model,
    dataset,
    device: str = 'cuda',
    max_samples: int = 500,
    method: str = 'tsne',
    ax=None
):
    """
    Visualize graph embeddings in 2D space.
    
    Parameters
    ----------
    model : torch.nn.Module
        Trained model (should have get_embedding method or we use penultimate layer)
    dataset : Dataset
        PyTorch Geometric dataset
    device : str
        Device
    max_samples : int
        Maximum samples to visualize
    method : str
        Dimensionality reduction method ('tsne' or 'umap')
    ax : matplotlib axes, optional
        Axes to plot on
        
    Returns
    -------
    matplotlib axes
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    model.eval()
    embeddings = []
    labels = []
    
    n_samples = min(len(dataset), max_samples)
    
    # Hook to capture embeddings before classifier
    embedding_output = []
    
    def hook_fn(module, input, output):
        embedding_output.append(output.detach())
    hook = None
    last_hookable = None
    hook_registered = False
    for name, module in model.named_modules():
        # Track the last layer before classifier
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            last_hookable = module
        if 'classifier' in name.lower() or 'fc' in name.lower() or 'mlp' in name.lower():
            # Found classifier, register hook on previous layer
            if last_hookable is not None:
                hook = last_hookable.register_forward_hook(hook_fn)
                hook_registered = True
            break
    if not hook_registered and not hasattr(model, 'forward_embedding'):
        warnings.warn("Could not register embedding hook and model has no forward_embedding method. "
                     "Embeddings will be extracted from model output directly, which may not be optimal.")
    
    # Use forward_embedding if available, otherwise use hook or raw output
    with torch.no_grad():
        for i in range(n_samples):
            data = dataset[i].to(device)
            embedding_output.clear()  # Clear previous hook output
            
            # Try forward_embedding method first (most reliable)
            if hasattr(model, 'forward_embedding'):
                emb = model.forward_embedding(data)
            else:
                # Run forward pass (hook will capture if registered)
                output = model(data)
                
                if embedding_output:
                    # Use hooked embedding
                    emb = embedding_output[-1]
                elif output.dim() == 0:
                    emb = torch.tensor([output.item()])
                else:
                    emb = output.flatten()
            
            embeddings.append(emb.cpu().numpy().flatten())
            if hasattr(data, 'y'):
                labels.append(int(data.y.item()))
    
    # Remove hook if registered
    if hook is not None:
        hook.remove()
    
    embeddings = np.array(embeddings)
    labels = np.array(labels) if labels else None
    
    # Dimensionality reduction
    if embeddings.shape[1] > 2:
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
        else:
            try:
                import umap
                reducer = umap.UMAP(n_components=2, random_state=42)
            except ImportError:
                warnings.warn("UMAP not available, using t-SNE")
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
        
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings
    
    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        scatter = ax.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=labels, cmap='coolwarm', alpha=0.6
        )
        plt.colorbar(scatter, ax=ax, label='Class')
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(f'Graph Embedding Space ({method.upper()})')
    
    return ax


def save_interpretation_results(
    results: Dict,
    output_path: Path,
    include_raw_importance: bool = False
):
    """
    Save interpretation results to JSON.
    
    Parameters
    ----------
    results : Dict
        Interpretation results
    output_path : Path
        Output file path
    include_raw_importance : bool
        Whether to include full importance arrays (can be large)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Clean results for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, tuple):
            return [clean_for_json(v) for v in obj]
        else:
            return obj
    
    results_clean = clean_for_json(results)
    
    # Optionally remove large arrays
    if not include_raw_importance:
        if 'node_importance' in results_clean:
            del results_clean['node_importance']
        if isinstance(results_clean, list):
            for r in results_clean:
                if isinstance(r, dict) and 'node_importance' in r:
                    del r['node_importance']
    
    results_clean['timestamp'] = datetime.now().isoformat()
    
    with open(output_path, 'w') as f:
        json.dump(results_clean, f, indent=2)


if __name__ == "__main__":
    print("Model interpretation module loaded successfully")
    print("Available functions:")
    print("  - compute_node_gradients")
    print("  - compute_integrated_gradients")
    print("  - get_top_important_nodes")
    print("  - aggregate_importance_across_samples")
    print("  - analyze_prediction")
    print("  - batch_analyze_predictions")
    print("  - visualize_node_importance")
    print("  - visualize_embedding_space")
    print("  - save_interpretation_results")

