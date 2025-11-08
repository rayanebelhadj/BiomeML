#!/usr/bin/env python3
"""
GNN models for microbiome disease classification.
Different architectures (GCN, GINEConv, GAT, GraphSAGE) + baseline models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv, GATConv, SAGEConv, GCNConv, RGCNConv,
    global_mean_pool, global_max_pool, global_add_pool,
    BatchNorm
)
from torch.nn import Linear, Sequential, LeakyReLU, Sigmoid
from typing import Optional
import numpy as np


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classes. FL = -alpha * (1-p)^gamma * log(p)"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = inputs
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedBCELoss(nn.Module):
    """Weighted BCE for class imbalance - computes weights from data if not given."""
    
    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        self.pos_weight = pos_weight
    
    @staticmethod
    def compute_pos_weight(labels: np.ndarray) -> float:
        """Compute positive class weight from labels."""
        n_pos = np.sum(labels)
        n_neg = len(labels) - n_pos
        return n_neg / n_pos if n_pos > 0 else 1.0
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # The previous code applied logit() then binary_cross_entropy_with_logits() which
        # applies sigmoid internally, creating redundant computation and numeric instability
        if self.pos_weight is not None:
            pos_weight = torch.tensor([self.pos_weight], device=inputs.device)
            # Use weighted BCE directly on sigmoid outputs (inputs are already sigmoid)
            inputs_clamped = inputs.clamp(1e-7, 1 - 1e-7)
            # Manual weighted BCE: -w*y*log(p) - (1-y)*log(1-p)
            bce = -pos_weight * targets * torch.log(inputs_clamped) - (1 - targets) * torch.log(1 - inputs_clamped)
            return bce.mean()
        else:
            return F.binary_cross_entropy(inputs.clamp(1e-7, 1 - 1e-7), targets)


class MLP_Baseline(nn.Module):
    """
    MLP baseline - no graph structure, just pooled node features.
    Used as control to check if graph structure actually helps.
    """
    
    def __init__(
        self, 
        input_dim: int = 1, 
        hidden_dim: int = 128, 
        num_layers: int = 3, 
        dropout: float = 0.3, 
        pooling: str = 'mean',
        num_classes: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pooling = pooling
        self.num_classes = num_classes
        
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        
        # MLP layers - no message passing
        layers = []
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()
        
        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
    
    def forward(self, data):
        x, batch = data.x, data.batch
        
        x = self.node_encoder(x)
        
        # no message passing here, just node-wise MLP
        if not isinstance(self.mlp, nn.Identity):
            x = self.mlp(x)
        
        # pool
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        
        out = self.classifier(x)
        
        if self.num_classes == 2:
            return out.squeeze(-1)
        return out
    
    def forward_embedding(self, data):
        """Get graph embedding before classifier."""
        x, batch = data.x, data.batch
        
        x = self.node_encoder(x)
        
        if not isinstance(self.mlp, nn.Identity):
            x = self.mlp(x)
        
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        
        return x


class GNN_GCN(nn.Module):
    """
    Basic GCN - uses graph structure but NOT edge features.
    Good baseline for comparison.
    """
    
    def __init__(
        self, 
        input_dim: int = 1, 
        hidden_dim: int = 128, 
        num_layers: int = 2, 
        dropout: float = 0.3, 
        pooling: str = 'mean',
        num_classes: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pooling = pooling
        self.num_classes = num_classes
        
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.node_encoder(x)
        
        # GCN doesnt use edge_attr
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        
        out = self.classifier(x)
        
        if self.num_classes == 2:
            return out.squeeze(-1)
        return out
    
    def forward_embedding(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.node_encoder(x)
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        
        return x


class GNN_GINEConv(nn.Module):
    """GIN with edge features - more expressive, uses phylogenetic distances."""
    
    def __init__(
        self, 
        input_dim: int = 1, 
        hidden_dim: int = 128, 
        num_layers: int = 2, 
        dropout: float = 0.3, 
        pooling: str = 'mean',
        num_classes: int = 2
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pooling = pooling
        self.num_classes = num_classes
        
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.edge_encoder = nn.Linear(1, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(nn=mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        
        out = self.classifier(x)
        
        if self.num_classes == 2:
            return out.squeeze(-1)
        return out
    
    def forward_embedding(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        
        return x


class EdgeCentricRGCN(nn.Module):
    """
    Edge-centric model based on professor's reference notebooks.
    Uses GINEConv with node weights (abundance) and edge weights (distance).
    """
    
    def __init__(
        self, 
        input_dim: int = 1,  # Added for API compatibility
        hidden_dim: int = 128, 
        num_layers: int = 2,  # Added for API compatibility (uses fixed 2 layers)
        num_classes: int = 2,
        dropout: float = 0.3,
        pooling: str = 'mean'
    ):
        super().__init__()
        self.input_dim = input_dim  # Store for compatibility
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers  # Store but use fixed architecture
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.pooling = pooling
        
        self.node_encoder = Linear(1, hidden_dim)
        self.edge_encoder = Linear(1, hidden_dim)

        self.conv1 = GINEConv(
            nn=Sequential(
                Linear(hidden_dim, hidden_dim),
                LeakyReLU(),
                Linear(hidden_dim, hidden_dim)
            ))

        self.conv2 = GINEConv(
            nn=Sequential(
                Linear(hidden_dim, hidden_dim),
                LeakyReLU(),
                Linear(hidden_dim, hidden_dim)
            ))

        if num_classes == 2:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                Sigmoid()
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            )

    def forward(self, data):
        edge_index = data.edge_index
        batch = data.batch
        device = edge_index.device
        
        # handle different attribute names
        if hasattr(data, 'node_weight'):
            node_weight = data.node_weight.to(device)
        else:
            node_weight = data.x.squeeze(-1).to(device)
        
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            edge_weight = data.edge_weight.to(device)
        else:
            edge_weight = data.edge_attr.squeeze(-1).to(device)  # Use defined encoders
        node_weight = node_weight.unsqueeze(-1) if node_weight.dim() == 1 else node_weight
        x = self.node_encoder(node_weight)
        
        edge_weight = edge_weight.unsqueeze(-1) if edge_weight.dim() == 1 else edge_weight
        edge_attr = self.edge_encoder(edge_weight)
        
        # conv layers
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # pool
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        
        out = self.mlp(x)
        
        if self.num_classes == 2:
            return out.squeeze(-1)
        return out
    
    def forward_embedding(self, data):
        edge_index = data.edge_index
        batch = data.batch
        device = edge_index.device
        
        if hasattr(data, 'node_weight'):
            node_weight = data.node_weight.to(device)
        else:
            node_weight = data.x.squeeze(-1).to(device)
        
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            edge_weight = data.edge_weight.to(device)
        else:
            edge_weight = data.edge_attr.squeeze(-1).to(device)
        
        node_weight = node_weight.unsqueeze(-1) if node_weight.dim() == 1 else node_weight
        x = self.node_encoder(node_weight)
        
        edge_weight = edge_weight.unsqueeze(-1) if edge_weight.dim() == 1 else edge_weight
        edge_attr = self.edge_encoder(edge_weight)
        
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        
        return x


class GNN_GAT(nn.Module):
    """
    Graph attention network - learns which edges matter via attention.
    FIXED: Added skip connections, removed double dropout, fixed forward_embedding.
    """
    
    def __init__(
        self, 
        input_dim: int = 1, 
        hidden_dim: int = 128, 
        num_layers: int = 2, 
        dropout: float = 0.3, 
        pooling: str = 'mean',
        num_classes: int = 2,
        heads: int = 4,
        edge_dim: int = 1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pooling = pooling
        self.num_classes = num_classes
        self.heads = heads
        
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.edge_encoder = nn.Linear(1, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # FIX: Remove dropout from GATConv (use external dropout only)
            # FIX: Use add_self_loops=False since we handle it ourselves
            self.convs.append(GATConv(
                hidden_dim, 
                hidden_dim // heads, 
                heads=heads,
                dropout=0.0,  # FIX: No internal dropout (was causing double dropout)
                edge_dim=hidden_dim,
                concat=True,
                add_self_loops=False  # FIX: Explicit control
            ))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.node_encoder(x)
        edge_attr_encoded = self.edge_encoder(edge_attr)
        
        for conv, bn in zip(self.convs, self.batch_norms):
            # FIX: Add skip connection (residual) for better gradient flow
            identity = x
            x = conv(x, edge_index, edge_attr=edge_attr_encoded)
            x = bn(x)
            x = F.relu(x)
            x = x + identity  # FIX: Skip connection
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        
        out = self.classifier(x)
        
        if self.num_classes == 2:
            return out.squeeze(-1)
        return out
    
    def forward_embedding(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.node_encoder(x)
        # FIX: Was missing edge encoding in forward_embedding!
        edge_attr_encoded = self.edge_encoder(edge_attr)
        
        for conv, bn in zip(self.convs, self.batch_norms):
            identity = x
            x = conv(x, edge_index, edge_attr=edge_attr_encoded)
            x = bn(x)
            x = F.relu(x)
            x = x + identity  # Skip connection
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        
        return x


class GNN_GraphSAGE(nn.Module):
    """GraphSAGE - good for inductive learning on new graphs."""
    
    def __init__(
        self, 
        input_dim: int = 1, 
        hidden_dim: int = 128, 
        num_layers: int = 2, 
        dropout: float = 0.3, 
        pooling: str = 'mean',
        num_classes: int = 2,
        aggr: str = 'mean'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pooling = pooling
        self.num_classes = num_classes
        
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.node_encoder(x)
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        
        out = self.classifier(x)
        
        if self.num_classes == 2:
            return out.squeeze(-1)
        return out

    def forward_embedding(self, data):
        """Get graph-level embeddings before classification."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.node_encoder(x)
        
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        
        return x


class GNN_Ensemble(nn.Module):
    """Ensemble combining GAT + GraphSAGE."""
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        pooling: str = 'mean',
        num_classes: int = 2
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        from torch_geometric.nn import GINEConv
        
        self.gat = GNN_GAT(input_dim, hidden_dim, num_layers, dropout, pooling, num_classes)
        self.sage = GNN_GraphSAGE(input_dim, hidden_dim, num_layers, dropout, pooling, num_classes)
        
        self.combination = nn.Linear(2, 1) if num_classes == 2 else nn.Linear(2 * num_classes, num_classes)
    
    def forward(self, data):
        gat_out = self.gat(data)
        sage_out = self.sage(data)
        
        if self.num_classes == 2:
            combined = torch.stack([gat_out, sage_out], dim=-1)
            out = torch.sigmoid(self.combination(combined).squeeze(-1))
            return out
        else:
            combined = torch.cat([gat_out, sage_out], dim=-1)
            return self.combination(combined)


class CNN_Baseline(nn.Module):
    """
    1D CNN baseline - treats abundance as a 1D signal.
    Tests if graphs are better than fixed CNN neighborhoods.
    """
    
    def __init__(
        self,
        input_dim: int = 1000,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(128, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(128, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes)
            )
    
    def forward(self, data):
        """Accepts PyG data for interface compatibility."""
        x, batch = data.x, data.batch
        
        # reconstruct per-graph feature vectors
        unique_batches = batch.unique()
        batch_size = len(unique_batches)
        
        graph_features = []
        for b in unique_batches:
            mask = (batch == b)
            graph_x = x[mask].squeeze(-1)
            graph_features.append(graph_x)
        
        # pad and stack
        max_len = max(len(gf) for gf in graph_features)
        padded = torch.zeros(batch_size, max_len, device=x.device)
        for i, gf in enumerate(graph_features):
            padded[i, :len(gf)] = gf
        
        x = padded.unsqueeze(1)  # (batch, 1, seq_len)
        
        # conv forward
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = self.global_pool(x).squeeze(-1)
        
        out = self.classifier(x)
        
        if self.num_classes == 2:
            return out.squeeze(-1)
        return out
    
    def forward_embedding(self, data):
        """"""
        x, batch = data.x, data.batch
        
        # reconstruct per-graph feature vectors
        unique_batches = batch.unique()
        batch_size = len(unique_batches)
        
        graph_features = []
        for b in unique_batches:
            mask = (batch == b)
            graph_x = x[mask].squeeze(-1)
            graph_features.append(graph_x)
        
        # pad and stack
        max_len = max(len(gf) for gf in graph_features)
        padded = torch.zeros(batch_size, max_len, device=x.device)
        for i, gf in enumerate(graph_features):
            padded[i, :len(gf)] = gf
        
        x = padded.unsqueeze(1)  # (batch, 1, seq_len)
        
        # conv forward (without classifier)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x).squeeze(-1)
        
        return x


class GNN_WithMetadata(nn.Module):
    """
    GNN + patient metadata (age, sex, bmi, antibiotics) via late fusion.
    Graph embedding concatenated with encoded metadata before classifier.
    """
    
    def __init__(
        self,
        gnn_type: str = 'GINEConv',
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        pooling: str = 'mean',
        num_classes: int = 2,
        meta_hidden_dim: int = 32
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.meta_hidden_dim = meta_hidden_dim
        
        gnn_classes = {
            'GCN': GNN_GCN,
            'GINEConv': GNN_GINEConv,
            'GAT': GNN_GAT,
            'GraphSAGE': GNN_GraphSAGE,
        }
        
        if gnn_type not in gnn_classes:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        self.gnn = gnn_classes[gnn_type](
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            pooling=pooling,
            num_classes=num_classes
        )
        
        # metadata encoder
        self.meta_encoder = nn.Sequential(
            nn.Linear(4, meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(meta_hidden_dim, meta_hidden_dim)
        )
        
        combined_dim = hidden_dim + meta_hidden_dim
        
        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, combined_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(combined_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, combined_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(combined_dim // 2, num_classes)
            )
    
    def forward(self, data, metadata: torch.Tensor):
        graph_emb = self.gnn.forward_embedding(data)
        meta_emb = self.meta_encoder(metadata)
        combined = torch.cat([graph_emb, meta_emb], dim=-1)
        out = self.classifier(combined)
        
        if self.num_classes == 2:
            return out.squeeze(-1)
        return out
    
    def forward_graph_only(self, data):
        """Forward using only graph (for comparison)."""
        return self.gnn(data)


# This class is expected by the cross-validation code when clinical features are enabled
GNN_Clinical = GNN_WithMetadata


def get_model(model_type: str, **kwargs):
    """Factory function - get model by name."""
    models = {
        'GCN': GNN_GCN,
        'GINEConv': GNN_GINEConv,
        'GAT': GNN_GAT,
        'GraphSAGE': GNN_GraphSAGE,
        'EdgeCentricRGCN': EdgeCentricRGCN,
        'Ensemble': GNN_Ensemble,
        'MLP': MLP_Baseline,
        'CNN': CNN_Baseline,
        'GNN_Meta': GNN_WithMetadata,
        'GNN_Clinical': GNN_Clinical,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](**kwargs)


def get_loss(loss_type: str, **kwargs):
    """Factory function for loss functions."""
    if loss_type == 'bce':
        return nn.BCELoss()
    elif loss_type == 'focal':
        alpha = kwargs.get('alpha', 0.25)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_type == 'weighted_bce':
        pos_weight = kwargs.get('pos_weight', None)
        return WeightedBCELoss(pos_weight=pos_weight)
    else:
        raise ValueError(f"Unknown loss: {loss_type}")


if __name__ == "__main__":
    print("Testing models...")
    
    from torch_geometric.data import Data, Batch
    
    x = torch.randn(10, 1)
    edge_index = torch.tensor([[0,1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9,0]], dtype=torch.long)
    edge_attr = torch.randn(10, 1)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch = Batch.from_data_list([data, data])
    
    # test MLP
    mlp = MLP_Baseline(input_dim=1, hidden_dim=64, num_layers=3)
    out = mlp(batch)
    print(f"MLP output: {out.shape}")
    
    # test GAT
    gat = GNN_GAT(input_dim=1, hidden_dim=64, num_layers=2)
    out = gat(batch)
    print(f"GAT output: {out.shape}")
    
    # test focal loss
    focal = FocalLoss()
    pred = torch.sigmoid(torch.randn(10))
    target = torch.randint(0, 2, (10,)).float()
    loss = focal(pred, target)
    print(f"Focal loss: {loss.item():.4f}")
    
    print("All tests passed")
