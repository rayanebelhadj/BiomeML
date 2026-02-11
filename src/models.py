#!/usr/bin/env python3
"""
GNN models for microbiome disease classification.
Different architectures (GCN, GINEConv, GAT, GraphSAGE) + baseline models.

Output interface:
  - num_classes=2 (binary): forward() returns probabilities in [0, 1], shape [batch]
  - num_classes>2 (multiclass): forward() returns logits (unnormalized), shape [batch, num_classes]
  - BCE / Focal / WeightedBCE expect probabilities (binary)
  - CrossEntropyLoss expects logits (multiclass)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv, GATConv, SAGEConv, GCNConv,
    global_mean_pool, global_max_pool, global_add_pool,
)
from torch.nn import Linear, Sequential, LeakyReLU, Sigmoid
from typing import Optional
import numpy as np


_EDGE_ATTR_WARNED = False


def _warn_missing_edge_attr():
    """Emit a one-time warning when edge_attr is missing from a Data object."""
    global _EDGE_ATTR_WARNED
    if not _EDGE_ATTR_WARNED:
        import warnings
        warnings.warn(
            "data.edge_attr is None — using ones (all edges weighted equally). "
            "This is normal for GCN/GraphSAGE but may indicate a bug for "
            "GINEConv/GAT which expect phylogenetic distance as edge features.",
            RuntimeWarning,
            stacklevel=4,
        )
        _EDGE_ATTR_WARNED = True


def _validate_pooling(pooling: str) -> None:
    """Raise ValueError if pooling is not one of the supported values."""
    if pooling not in ('mean', 'max', 'sum'):
        raise ValueError(f"pooling must be 'mean', 'max', or 'sum', got {pooling!r}")


def _safe_hidden_dim_div2(h: int) -> int:
    """Return h//2 ensuring at least 1 to avoid Linear(in_dim, 0)."""
    return max(1, h // 2)


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classes. FL = -alpha_t * (1-p_t)^gamma * log(p_t) with p_t and alpha_t for class balancing."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"FocalLoss: reduction must be 'mean', 'sum', or 'none', got {reduction!r}")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs_clamped = inputs.clamp(1e-7, 1 - 1e-7)
        ce_loss = F.binary_cross_entropy(inputs_clamped, targets, reduction='none')
        p_t = inputs_clamped * targets + (1 - inputs_clamped) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class BCELossClamped(nn.Module):
    """BCE loss with input clamping for numerical stability (consistent with FocalLoss/WeightedBCELoss)."""

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy(inputs.clamp(1e-7, 1 - 1e-7), targets)


class WeightedBCELoss(nn.Module):
    """Weighted BCE for class imbalance. Pass pos_weight=compute_pos_weight(labels) to enable weighting; otherwise uses standard BCE."""
    
    def __init__(self, pos_weight: Optional[float] = None):
        super().__init__()
        self.pos_weight = pos_weight
    
    @staticmethod
    def compute_pos_weight(labels: np.ndarray) -> float:
        """Compute positive class weight from labels. Returns 1.0 if n_neg=0 or n_pos=0 to avoid degenerate loss."""
        n_pos = int(np.sum(labels))
        n_neg = len(labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return 1.0  # Avoid degenerate pos_weight=0 when all samples are positive
        return n_neg / n_pos
    
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
        _validate_pooling(pooling)
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
        
        mid = _safe_hidden_dim_div2(hidden_dim)
        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, mid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mid, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, mid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mid, num_classes)
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
    """Basic GCN - uses graph structure but NOT edge features."""
    
    def __init__(
        self, 
        input_dim: int = 1, 
        hidden_dim: int = 128, 
        num_layers: int = 2, 
        dropout: float = 0.3, 
        pooling: str = 'mean',
        num_classes: int = 2,
        include_classifier: bool = True
    ):
        super().__init__()
        _validate_pooling(pooling)
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
        
        mid = _safe_hidden_dim_div2(hidden_dim)
        if include_classifier:
            if num_classes == 2:
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, mid),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mid, 1),
                    nn.Sigmoid()
                )
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, mid),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mid, num_classes)
                )
        else:
            # Minimal classifier for GNN_WithMetadata (uses forward_embedding; forward_graph_only needs this)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 1 if num_classes == 2 else num_classes),
                nn.Sigmoid() if num_classes == 2 else nn.Identity()
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
    """GINEConv - edge-featured GIN for phylogenetic graphs."""
    
    def __init__(
        self, 
        input_dim: int = 1, 
        hidden_dim: int = 128, 
        num_layers: int = 2, 
        dropout: float = 0.3, 
        pooling: str = 'mean',
        num_classes: int = 2,
        include_classifier: bool = True
    ):
        super().__init__()
        _validate_pooling(pooling)
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
        
        mid = _safe_hidden_dim_div2(hidden_dim)
        if include_classifier:
            if num_classes == 2:
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, mid),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mid, 1),
                    nn.Sigmoid()
                )
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, mid),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mid, num_classes)
                )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 1 if num_classes == 2 else num_classes),
                nn.Sigmoid() if num_classes == 2 else nn.Identity()
            )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is None:
            _warn_missing_edge_attr()
            edge_attr = torch.ones(edge_index.shape[1], 1, device=x.device, dtype=x.dtype)
        elif edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
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
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is None:
            _warn_missing_edge_attr()
            edge_attr = torch.ones(edge_index.shape[1], 1, device=x.device, dtype=x.dtype)
        elif edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
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
    Edge-centric model inspired by professor's reference notebooks.
    Uses two GINEConv layers with separate node/edge encoders.
    Note: num_layers is stored for API compatibility but architecture is fixed at 2 layers.
    """

    def __init__(
        self,
        input_dim: int = 1,  # Added for API compatibility
        hidden_dim: int = 128,
        num_layers: int = 2,  # Stored for API; architecture fixed at 2 layers
        num_classes: int = 2,
        dropout: float = 0.3,
        pooling: str = 'mean',
        edge_dim: int = 1
    ):
        super().__init__()
        _validate_pooling(pooling)
        if num_layers != 2:
            import warnings
            warnings.warn(
                f"EdgeCentricRGCN: num_layers={num_layers} ignored; architecture uses fixed 2 layers.",
                UserWarning,
                stacklevel=2
            )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = 2
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.pooling = pooling

        self.node_encoder = Linear(input_dim, hidden_dim)
        self.edge_encoder = Linear(edge_dim, hidden_dim)

        mlp1 = Sequential(
            Linear(hidden_dim, hidden_dim),
            LeakyReLU(),
            Linear(hidden_dim, hidden_dim))
        self.conv1 = GINEConv(nn=mlp1, train_eps=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        mlp2 = Sequential(
            Linear(hidden_dim, hidden_dim),
            LeakyReLU(),
            Linear(hidden_dim, hidden_dim))
        self.conv2 = GINEConv(nn=mlp2, train_eps=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

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
        
        # handle different attribute names; expects data.x shape [N, 1] or [N]
        if hasattr(data, 'node_weight'):
            node_weight = data.node_weight.to(device)
        else:
            if data.x.dim() > 1 and data.x.size(-1) > 1:
                raise ValueError(f"EdgeCentricRGCN expects data.x shape [N, 1], got {data.x.shape}")
            node_weight = data.x.squeeze(-1).to(device)
        
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            edge_weight = data.edge_weight.to(device)
        elif hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_weight = data.edge_attr.to(device)
        else:
            n_edges = edge_index.shape[1]
            edge_weight = torch.ones(n_edges, device=device)
        node_weight = node_weight.unsqueeze(-1) if node_weight.dim() == 1 else node_weight
        x = self.node_encoder(node_weight)

        # Ensure edge_weight has shape [E, edge_dim]; use first column if multi-dim
        if edge_weight.dim() == 1:
            edge_weight = edge_weight.unsqueeze(-1)
        elif edge_weight.size(-1) > 1:
            edge_weight = edge_weight[:, : self.edge_encoder.in_features]
        edge_attr = self.edge_encoder(edge_weight)

        # conv layers with BatchNorm
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
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
            if data.x.dim() > 1 and data.x.size(-1) > 1:
                raise ValueError(f"EdgeCentricRGCN expects data.x shape [N, 1], got {data.x.shape}")
            node_weight = data.x.squeeze(-1).to(device)
        
        if hasattr(data, 'edge_weight') and data.edge_weight is not None:
            edge_weight = data.edge_weight.to(device)
        elif hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_weight = data.edge_attr.to(device)
        else:
            n_edges = edge_index.shape[1]
            edge_weight = torch.ones(n_edges, device=device)
        
        node_weight = node_weight.unsqueeze(-1) if node_weight.dim() == 1 else node_weight
        x = self.node_encoder(node_weight)
        
        if edge_weight.dim() == 1:
            edge_weight = edge_weight.unsqueeze(-1)
        elif edge_weight.size(-1) > 1:
            edge_weight = edge_weight[:, : self.edge_encoder.in_features]
        edge_attr = self.edge_encoder(edge_weight)
        
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
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
    Uses skip connections and add_self_loops (GAT standard).
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
        edge_dim: int = 1,
        include_classifier: bool = True
    ):
        super().__init__()
        _validate_pooling(pooling)
        if hidden_dim % heads != 0:
            raise ValueError(f"GNN_GAT: hidden_dim ({hidden_dim}) must be divisible by heads ({heads})")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.pooling = pooling
        self.num_classes = num_classes
        self.heads = heads
        
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # FIX: Remove dropout from GATConv (use external dropout only)
            # Use add_self_loops=True so each node includes its own features in attention (GAT standard)
            self.convs.append(GATConv(
                hidden_dim, 
                hidden_dim // heads, 
                heads=heads,
                dropout=0.0,  # No internal dropout (use external dropout only)
                edge_dim=hidden_dim,
                concat=True,
                add_self_loops=True
            ))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        mid = _safe_hidden_dim_div2(hidden_dim)
        if include_classifier:
            if num_classes == 2:
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, mid),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mid, 1),
                    nn.Sigmoid()
                )
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, mid),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mid, num_classes)
                )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 1 if num_classes == 2 else num_classes),
                nn.Sigmoid() if num_classes == 2 else nn.Identity()
            )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is None:
            _warn_missing_edge_attr()
            edge_attr = torch.ones(edge_index.shape[1], 1, device=x.device, dtype=x.dtype)
        elif edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        x = self.node_encoder(x)
        edge_attr_encoded = self.edge_encoder(edge_attr)
        
        for conv, bn in zip(self.convs, self.batch_norms):
            identity = x
            x = conv(x, edge_index, edge_attr=edge_attr_encoded)
            x = bn(x)
            x = F.relu(x)
            x = x + identity
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
        edge_attr = getattr(data, 'edge_attr', None)
        if edge_attr is None:
            _warn_missing_edge_attr()
            edge_attr = torch.ones(edge_index.shape[1], 1, device=x.device, dtype=x.dtype)
        elif edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        x = self.node_encoder(x)
        edge_attr_encoded = self.edge_encoder(edge_attr)
        
        for conv, bn in zip(self.convs, self.batch_norms):
            identity = x
            x = conv(x, edge_index, edge_attr=edge_attr_encoded)
            x = bn(x)
            x = F.relu(x)
            x = x + identity
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        if self.pooling == 'mean':
            x = global_mean_pool(x, batch)
        elif self.pooling == 'max':
            x = global_max_pool(x, batch)
        elif self.pooling == 'sum':
            x = global_add_pool(x, batch)
        
        return x


class GNN_GraphSAGE(nn.Module):
    """GraphSAGE - inductive GNN without edge features."""
    
    def __init__(
        self, 
        input_dim: int = 1, 
        hidden_dim: int = 128, 
        num_layers: int = 2, 
        dropout: float = 0.3, 
        pooling: str = 'mean',
        num_classes: int = 2,
        aggr: str = 'mean',
        include_classifier: bool = True
    ):
        super().__init__()
        _validate_pooling(pooling)
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
        
        mid = _safe_hidden_dim_div2(hidden_dim)
        if include_classifier:
            if num_classes == 2:
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, mid),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mid, 1),
                    nn.Sigmoid()
                )
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, mid),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(mid, num_classes)
                )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 1 if num_classes == 2 else num_classes),
                nn.Sigmoid() if num_classes == 2 else nn.Identity()
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
        _validate_pooling(pooling)
        if hidden_dim % 4 != 0:
            raise ValueError(
                f"GNN_Ensemble: hidden_dim ({hidden_dim}) must be divisible by 4 "
                "(GAT uses heads=4). Use hidden_dim=64, 128, 256, etc."
            )
        self.num_classes = num_classes
        
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
    
    def forward_embedding(self, data):
        """Concatenate GAT and GraphSAGE embeddings for interpretation (t-SNE/UMAP)."""
        gat_emb = self.gat.forward_embedding(data)
        sage_emb = self.sage.forward_embedding(data)
        return torch.cat([gat_emb, sage_emb], dim=-1)


class CNN_Baseline(nn.Module):
    """
    1D CNN baseline - treats abundance as a 1D signal.
    Tests if graphs are better than fixed CNN neighborhoods.
    Expects data.x shape [N, 1] (one scalar per node).
    Note: input_dim is stored for API compatibility only; sequence length is computed from data.x per batch.
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
        
        mid = _safe_hidden_dim_div2(hidden_dim)
        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(128, mid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mid, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(128, mid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mid, num_classes)
            )
    
    def forward(self, data):
        """Accepts PyG data for interface compatibility. Expects data.x shape [N, 1] (one scalar per node)."""
        x, batch = data.x, data.batch
        if x.dim() > 1 and x.size(-1) > 1:
            raise ValueError(f"CNN_Baseline expects data.x shape [N, 1], got {x.shape}")
        
        # reconstruct per-graph feature vectors (sorted=True for consistent order)
        unique_batches = batch.unique(sorted=True)
        batch_size = len(unique_batches)
        
        graph_features = []
        for b in unique_batches:
            mask = (batch == b)
            graph_x = x[mask].squeeze(-1)
            graph_features.append(graph_x)
        
        # pad and stack (guard: min length 4 to avoid MaxPool1d(2) producing dim 0)
        max_len = max(4, max(len(gf) for gf in graph_features))
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
        """Get graph embeddings before classifier (for model_interpretation). Expects data.x shape [N, 1]."""
        x, batch = data.x, data.batch
        if x.dim() > 1 and x.size(-1) > 1:
            raise ValueError(f"CNN_Baseline expects data.x shape [N, 1], got {x.shape}")
        
        # reconstruct per-graph feature vectors (sorted=True for consistent order)
        unique_batches = batch.unique(sorted=True)
        batch_size = len(unique_batches)
        
        graph_features = []
        for b in unique_batches:
            mask = (batch == b)
            graph_x = x[mask].squeeze(-1)
            graph_features.append(graph_x)
        
        # pad and stack (guard: min length 4 to avoid MaxPool1d(2) producing dim 0)
        max_len = max(4, max(len(gf) for gf in graph_features))
        padded = torch.zeros(batch_size, max_len, device=x.device)
        for i, gf in enumerate(graph_features):
            padded[i, :len(gf)] = gf
        
        x = padded.unsqueeze(1)  # (batch, 1, seq_len)
        
        # conv forward (without classifier) — dropout to match forward()
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        x = self.global_pool(x).squeeze(-1)
        
        return x


class MetadataOnly_Baseline(nn.Module):
    """
    Baseline using only clinical metadata (age, sex, BMI, etc.) - no graph.
    Tests if age/sex/BMI alone suffice to predict disease.
    Requires data.clinical [B, meta_dim].
    """
    
    def __init__(
        self,
        meta_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        self.meta_dim = meta_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        layers = []
        in_dim = meta_dim
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()
        
        mid = _safe_hidden_dim_div2(hidden_dim)
        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, mid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mid, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(in_dim, mid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mid, num_classes)
            )
    
    def forward(self, data):
        if not hasattr(data, 'clinical') or data.clinical is None:
            raise ValueError("MetadataOnly_Baseline requires data.clinical. Ensure use_clinical=True.")
        if data.clinical.shape[-1] != self.meta_dim:
            raise ValueError(
                f"MetadataOnly_Baseline: data.clinical shape {data.clinical.shape} does not match meta_dim={self.meta_dim}. "
                "Ensure meta_dim=clinical_dim when creating the model."
            )
        x = data.clinical
        x = self.encoder(x)
        out = self.classifier(x)
        if self.num_classes == 2:
            return out.squeeze(-1)
        return out
    
    def forward_embedding(self, data):
        """Get metadata embedding before classifier (for compatibility with model_interpretation)."""
        if not hasattr(data, 'clinical') or data.clinical is None:
            raise ValueError("MetadataOnly_Baseline requires data.clinical. Ensure use_clinical=True.")
        if data.clinical.shape[-1] != self.meta_dim:
            raise ValueError(
                f"MetadataOnly_Baseline: data.clinical shape {data.clinical.shape} does not match meta_dim={self.meta_dim}."
            )
        return self.encoder(data.clinical)


class GNN_WithMetadata(nn.Module):
    """
    GNN + patient metadata (age, sex, bmi, antibiotics) via late fusion.
    Graph embedding concatenated with encoded metadata before classifier.
    Reads data.clinical from PyG batch (expected shape [B, meta_dim]).
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
        meta_dim: int = 4,
        meta_hidden_dim: int = 32
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.meta_dim = meta_dim
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
            num_classes=num_classes,
            include_classifier=False  # We use forward_embedding only; minimal classifier for forward_graph_only
        )
        
        # metadata encoder
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(meta_hidden_dim, meta_hidden_dim)
        )
        
        combined_dim = hidden_dim + meta_hidden_dim
        mid = _safe_hidden_dim_div2(combined_dim)
        
        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, mid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mid, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(combined_dim, mid),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mid, num_classes)
            )
    
    def forward(self, data):
        if not hasattr(data, 'clinical') or data.clinical is None:
            raise ValueError("GNN_WithMetadata requires data.clinical. Ensure use_clinical=True and dataset provides clinical features.")
        if data.clinical.shape[-1] != self.meta_dim:
            raise ValueError(
                f"GNN_WithMetadata: data.clinical shape {data.clinical.shape} does not match meta_dim={self.meta_dim}. "
                "Ensure meta_dim=clinical_dim when creating the model."
            )
        graph_emb = self.gnn.forward_embedding(data)
        meta_emb = self.meta_encoder(data.clinical)
        combined = torch.cat([graph_emb, meta_emb], dim=-1)
        out = self.classifier(combined)
        
        if self.num_classes == 2:
            return out.squeeze(-1)
        return out
    
    def forward_embedding(self, data):
        """Get combined graph+metadata embedding for model_interpretation (t-SNE/UMAP)."""
        graph_emb = self.gnn.forward_embedding(data)
        if hasattr(data, 'clinical') and data.clinical is not None:
            meta_emb = self.meta_encoder(data.clinical)
            return torch.cat([graph_emb, meta_emb], dim=-1)
        # Fallback: zero-fill metadata embedding if clinical not available
        batch_size = graph_emb.size(0)
        meta_placeholder = torch.zeros(
            batch_size, self.meta_hidden_dim,
            device=graph_emb.device, dtype=graph_emb.dtype
        )
        return torch.cat([graph_emb, meta_placeholder], dim=-1)
    
    def forward_graph_only(self, data):
        """Forward using only the GNN (ignores metadata). Uses main classifier with zero-filled meta embedding."""
        graph_emb = self.gnn.forward_embedding(data)
        batch_size = graph_emb.size(0)
        meta_placeholder = torch.zeros(
            batch_size, self.meta_hidden_dim,
            device=graph_emb.device, dtype=graph_emb.dtype
        )
        combined = torch.cat([graph_emb, meta_placeholder], dim=-1)
        out = self.classifier(combined)
        if self.num_classes == 2:
            return out.squeeze(-1)
        return out


# This class is expected by the cross-validation code when clinical features are enabled
GNN_Clinical = GNN_WithMetadata


class SklearnModelWrapper(nn.Module):
    """
    PyG-compatible wrapper for sklearn models (RandomForest, XGBoost, etc.).
    extract_features computes 5 stats per graph: mean, std, max, min, n_nodes.
    parameters() returns a dummy param for optimizer compatibility.
    """
    
    def __init__(self, sklearn_model, num_classes: int = 2):
        super().__init__()
        self.sklearn_model = sklearn_model
        self.num_classes = num_classes
        self._dummy = nn.Parameter(torch.zeros(1))
    
    @staticmethod
    def extract_features(data) -> np.ndarray:
        """Extract 5 stats per graph: mean, std, max, min, n_nodes."""
        x, batch = data.x, data.batch
        unique_batches = batch.unique(sorted=True)
        features = []
        for b in unique_batches:
            mask = batch == b
            node_vals = x[mask].reshape(-1).detach().cpu().numpy()
            n_nodes = int(mask.sum().item())
            if len(node_vals) == 0:
                features.append([0.0, 0.0, 0.0, 0.0, n_nodes])
            else:
                features.append([
                    float(np.mean(node_vals)),
                    float(np.std(node_vals)) if len(node_vals) > 1 else 0.0,
                    float(np.max(node_vals)),
                    float(np.min(node_vals)),
                    n_nodes
                ])
        return np.array(features, dtype=np.float32)
    
    def fit_from_loader(self, loader):
        """Fit sklearn model on features extracted from all batches in loader."""
        X_list, y_list = [], []
        for batch in loader:
            X_list.append(self.extract_features(batch))
            y_list.append(batch.y.cpu().numpy())
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        self.sklearn_model.fit(X, y)
        return self
    
    def forward(self, data):
        """Return probabilities for num_classes=2 (BCELoss) or logits for num_classes>2 (CrossEntropyLoss)."""
        X = self.extract_features(data)
        probs = self.sklearn_model.predict_proba(X)
        device = data.x.device if hasattr(data, 'x') and data.x is not None else self._dummy.device
        if self.num_classes == 2:
            return torch.from_numpy(probs[:, 1]).float().to(device)
        # For multiclass: convert probs to logits for CrossEntropyLoss compatibility
        probs_clamped = np.clip(probs, 1e-7, 1 - 1e-7)
        probs_clamped = probs_clamped / probs_clamped.sum(axis=1, keepdims=True)  # Re-normalize after clip
        logits = np.log(probs_clamped).astype(np.float32)
        return torch.from_numpy(logits).float().to(device)

    def forward_embedding(self, data):
        """Return extracted features as embeddings for model_interpretation (t-SNE/UMAP)."""
        X = self.extract_features(data)
        device = data.x.device if hasattr(data, 'x') and data.x is not None else self._dummy.device
        return torch.from_numpy(X).float().to(device)


def get_model(model_type: str, **kwargs):
    """Factory function - get model by name. Filters kwargs to match each model's signature."""
    import inspect
    
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
        'MetadataOnly': MetadataOnly_Baseline,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model: {model_type}. Available: {list(models.keys())}")
    
    cls = models[model_type]
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self'}
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_keyword:
        filtered = kwargs
    else:
        filtered = {k: v for k, v in kwargs.items() if k in valid_params}
    
    return cls(**filtered)


def get_loss(loss_type: str, **kwargs):
    """Factory function for loss functions."""
    if loss_type == 'bce':
        return BCELossClamped()
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
    assert out.shape == (2,), f"MLP expected (2,), got {out.shape}"
    assert torch.isfinite(out).all(), "MLP output contains NaN/Inf"
    print(f"MLP output: {out.shape}")

    # test GAT
    gat = GNN_GAT(input_dim=1, hidden_dim=64, num_layers=2)
    out = gat(batch)
    assert out.shape == (2,), f"GAT expected (2,), got {out.shape}"
    assert (out >= 0).all() and (out <= 1).all(), "GAT output should be in [0, 1]"
    assert torch.isfinite(out).all(), "GAT output contains NaN/Inf"
    print(f"GAT output: {out.shape}")

    # test focal loss
    focal = FocalLoss()
    pred = torch.sigmoid(torch.randn(10))
    target = torch.randint(0, 2, (10,)).float()
    loss = focal(pred, target)
    assert torch.isfinite(loss).all(), "Focal loss is NaN/Inf"
    assert loss.dim() == 0, "Focal loss should be scalar"
    print(f"Focal loss: {loss.item():.4f}")

    # test CNN_Baseline with small graph (min 4 nodes for MaxPool)
    cnn_batch = Batch.from_data_list([Data(x=torch.randn(4, 1), edge_index=torch.zeros(2, 0).long(), y=torch.tensor([0.0]))])
    cnn = CNN_Baseline(input_dim=4, hidden_dim=64, num_classes=2)
    cnn_out = cnn(cnn_batch)
    assert cnn_out.shape == (1,), f"CNN expected (1,), got {cnn_out.shape}"
    assert torch.isfinite(cnn_out).all(), "CNN output contains NaN/Inf"
    print("CNN (4-node graph): OK")

    print("All tests passed")
