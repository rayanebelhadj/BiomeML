#!/usr/bin/env python3
"""GNN models for microbiome disease classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv, GATConv, SAGEConv, GCNConv,
    global_mean_pool, global_max_pool, global_add_pool,
)
from typing import Optional
import numpy as np


class GNN_GCN(nn.Module):
    """Basic GCN - uses graph structure but NOT edge features."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, pooling: str = 'mean', num_classes: int = 2):
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
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes)
            )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_encoder(x)
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if self.pooling == 'mean': x = global_mean_pool(x, batch)
        elif self.pooling == 'max': x = global_max_pool(x, batch)
        elif self.pooling == 'sum': x = global_add_pool(x, batch)
        out = self.classifier(x)
        return out.squeeze(-1) if self.num_classes == 2 else out

    def forward_embedding(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_encoder(x)
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index); x = bn(x); x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if self.pooling == 'mean': x = global_mean_pool(x, batch)
        elif self.pooling == 'max': x = global_max_pool(x, batch)
        elif self.pooling == 'sum': x = global_add_pool(x, batch)
        return x


class GNN_GAT(nn.Module):
    """Graph attention network - learns which edges matter via attention."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, pooling: str = 'mean', num_classes: int = 2,
                 heads: int = 4, edge_dim: int = 1):
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

        for _ in range(num_layers):
            # BUG: out_channels=hidden_dim makes output size hidden_dim*heads
            self.convs.append(GATConv(
                hidden_dim, hidden_dim,
                heads=heads,
                dropout=dropout,
                edge_dim=hidden_dim,
                concat=True,
                add_self_loops=False
            ))
            # BUG: BatchNorm expects hidden_dim but receives hidden_dim*heads
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        out_dim = hidden_dim * heads
        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(out_dim, hidden_dim // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(out_dim, hidden_dim // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes)
            )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)
        edge_attr_encoded = self.edge_encoder(edge_attr)
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr=edge_attr_encoded)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if self.pooling == 'mean': x = global_mean_pool(x, batch)
        elif self.pooling == 'max': x = global_max_pool(x, batch)
        elif self.pooling == 'sum': x = global_add_pool(x, batch)
        out = self.classifier(x)
        return out.squeeze(-1) if self.num_classes == 2 else out

    def forward_embedding(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)
        # BUG: missing edge encoding - forward_embedding uses raw edge_attr
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x); x = F.relu(x)
        if self.pooling == 'mean': x = global_mean_pool(x, batch)
        elif self.pooling == 'max': x = global_max_pool(x, batch)
        elif self.pooling == 'sum': x = global_add_pool(x, batch)
        return x


class GNN_GINEConv(nn.Module):
    """GIN with edge features - more expressive, uses phylogenetic distances."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, pooling: str = 'mean', num_classes: int = 2):
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
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(nn=mlp, train_eps=True))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        if num_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes)
            )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr); x = bn(x); x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if self.pooling == 'mean': x = global_mean_pool(x, batch)
        elif self.pooling == 'max': x = global_max_pool(x, batch)
        elif self.pooling == 'sum': x = global_add_pool(x, batch)
        out = self.classifier(x)
        return out.squeeze(-1) if self.num_classes == 2 else out

    def forward_embedding(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_encoder(x); edge_attr = self.edge_encoder(edge_attr)
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index, edge_attr); x = bn(x); x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if self.pooling == 'mean': x = global_mean_pool(x, batch)
        elif self.pooling == 'max': x = global_max_pool(x, batch)
        elif self.pooling == 'sum': x = global_add_pool(x, batch)
        return x


class GNN_GraphSAGE(nn.Module):
    """GraphSAGE - good for inductive learning on new graphs."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, pooling: str = 'mean', num_classes: int = 2,
                 aggr: str = 'mean'):
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
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes)
            )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_encoder(x)
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index); x = bn(x); x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if self.pooling == 'mean': x = global_mean_pool(x, batch)
        elif self.pooling == 'max': x = global_max_pool(x, batch)
        elif self.pooling == 'sum': x = global_add_pool(x, batch)
        out = self.classifier(x)
        return out.squeeze(-1) if self.num_classes == 2 else out

    def forward_embedding(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_encoder(x)
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, edge_index); x = bn(x); x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if self.pooling == 'mean': x = global_mean_pool(x, batch)
        elif self.pooling == 'max': x = global_max_pool(x, batch)
        elif self.pooling == 'sum': x = global_add_pool(x, batch)
        return x


def get_model(model_type: str, **kwargs):
    models = {'GCN': GNN_GCN, 'GAT': GNN_GAT, 'GINEConv': GNN_GINEConv, 'GraphSAGE': GNN_GraphSAGE}
    if model_type not in models:
        raise ValueError(f"Unknown model: {model_type}. Available: {list(models.keys())}")
    return models[model_type](**kwargs)


def get_loss(loss_type: str, **kwargs):
    if loss_type == 'bce':
        return nn.BCELoss()
    else:
        raise ValueError(f"Unknown loss: {loss_type}")
