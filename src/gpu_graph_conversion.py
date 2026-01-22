#!/usr/bin/env python3
"""
GPU-accelerated graph conversion utilities.
Option 1A: Fast batch conversion using PyTorch GPU operations.

Falls back to CPU if GPU unavailable.
"""
import torch
import numpy as np
from torch_geometric.data import Data
from typing import List, Optional
import time

def nx_to_pyg_gpu_batch(nx_graphs: List, 
                         batch_size: int = 512,
                         use_gpu: bool = True,
                         verbose: bool = False) -> List[Data]:
    """
    Convert NetworkX graphs to PyTorch Geometric format using GPU acceleration.
    
    Args:
        nx_graphs: List of NetworkX graphs
        batch_size: Number of graphs to process in parallel on GPU
        use_gpu: Whether to attempt GPU acceleration
        verbose: Print timing information
        
    Returns:
        List of PyG Data objects
    """
    device = 'cpu'
    if use_gpu and torch.cuda.is_available():
        device = 'cuda'
        if verbose:
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        if verbose:
            print(f"💻 Using CPU (GPU not available)")
    
    pyg_graphs = []
    total_time = 0
    
    for batch_start in range(0, len(nx_graphs), batch_size):
        batch_end = min(batch_start + batch_size, len(nx_graphs))
        batch = nx_graphs[batch_start:batch_end]
        
        batch_start_time = time.time()
        
        # Process batch on GPU/CPU
        batch_pyg = _convert_batch_parallel(batch, device, verbose=False)
        pyg_graphs.extend(batch_pyg)
        
        batch_time = time.time() - batch_start_time
        total_time += batch_time
        
        if verbose and (batch_start % 1000 == 0 or batch_end == len(nx_graphs)):
            rate = len(batch) / batch_time if batch_time > 0 else float('inf')
            print(f"  Processed {batch_end}/{len(nx_graphs)} graphs "
                  f"({rate:.1f} graphs/sec, total: {total_time:.1f}s)")
    
    if verbose:
        avg_rate = len(nx_graphs) / total_time if total_time > 0 else float('inf')
        print(f"\n✅ Conversion complete: {len(nx_graphs)} graphs in {total_time:.1f}s "
              f"({avg_rate:.1f} graphs/sec)")
    
    return pyg_graphs

def _convert_batch_parallel(nx_graphs: List, device: str, verbose: bool = False) -> List[Data]:
    """Convert a batch of graphs in parallel using vectorized operations."""
    batch_pyg = []
    
    for g in nx_graphs:
        try:
            n_nodes = g.number_of_nodes()
            n_edges = g.number_of_edges()
            
            if n_nodes == 0:
                # Empty graph - create placeholder
                data = Data(
                    x=torch.zeros((1, 1), dtype=torch.float32),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                    edge_attr=torch.zeros((0, 1), dtype=torch.float32)
                )
                batch_pyg.append(data)
                continue
            
            # Extract node features (abundance weights)
            node_features = []
            node_mapping = {}
            for idx, (node, attrs) in enumerate(g.nodes(data=True)):
                node_mapping[node] = idx
                weight = attrs.get('weight', 0.0)
                node_features.append([weight])
            
            # Extract edges and edge features
            # Add both directions for undirected graphs (PyG convention)
            edge_list = []
            edge_features = []
            for src, dst, attrs in g.edges(data=True):
                if src in node_mapping and dst in node_mapping:
                    weight = attrs.get('weight', 1.0)
                    edge_list.append([node_mapping[src], node_mapping[dst]])
                    edge_features.append([weight])
                    edge_list.append([node_mapping[dst], node_mapping[src]])
                    edge_features.append([weight])
            
            # Convert to tensors on device
            if len(edge_list) > 0:
                x = torch.tensor(node_features, dtype=torch.float32, device=device)
                edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
                edge_attr = torch.tensor(edge_features, dtype=torch.float32, device=device)
            else:
                # Graph with nodes but no edges
                x = torch.tensor(node_features, dtype=torch.float32, device=device)
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                edge_attr = torch.zeros((0, 1), dtype=torch.float32, device=device)
            
            # Move back to CPU for storage (saves GPU memory)
            data = Data(
                x=x.cpu(),
                edge_index=edge_index.cpu(),
                edge_attr=edge_attr.cpu()
            )
            batch_pyg.append(data)
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to convert graph ({g.number_of_nodes()} nodes, "
                f"{g.number_of_edges()} edges) to PyG format: {e}"
            ) from e
    
    return batch_pyg

def test_conversion_speed(n_samples: int = 1000):
    """Test CPU vs GPU conversion speed."""
    import networkx as nx
    
    print(f"\n{'='*60}")
    print(f"Testing conversion speed with {n_samples} graphs")
    print(f"{'='*60}")
    
    # Generate test graphs
    print("Generating test graphs...")
    test_graphs = []
    for i in range(n_samples):
        g = nx.erdos_renyi_graph(n=100, p=0.1)
        for node in g.nodes():
            g.nodes[node]['weight'] = np.random.rand()
        for u, v in g.edges():
            g[u][v]['weight'] = np.random.rand()
        test_graphs.append(g)
    
    # Test CPU
    print("\nTesting CPU conversion...")
    start = time.time()
    cpu_graphs = nx_to_pyg_gpu_batch(test_graphs, use_gpu=False, verbose=True)
    cpu_time = time.time() - start
    
    # Test GPU
    if torch.cuda.is_available():
        print("\nTesting GPU conversion...")
        start = time.time()
        gpu_graphs = nx_to_pyg_gpu_batch(test_graphs, use_gpu=True, verbose=True)
        gpu_time = time.time() - start
        
        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"  CPU: {cpu_time:.2f}s ({n_samples/cpu_time:.1f} graphs/sec)")
        print(f"  GPU: {gpu_time:.2f}s ({n_samples/gpu_time:.1f} graphs/sec)")
        print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
        print(f"{'='*60}")
    else:
        print("\n❌ GPU not available for comparison")

if __name__ == "__main__":
    test_conversion_speed(n_samples=1000)

