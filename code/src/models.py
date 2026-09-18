"""Model definitions.

Baselines and target models from the surveyed papers:
- GCN: standard baseline (Kipf & Welling 2017)
- SemiGNN: Wang et al., ICDM 2019 (arXiv:2003.01171)
- Planned: CARE-GNN-style camouflage-resistant aggregation (Dou et al., CIKM 2020)
  and PC-GNN-style balanced neighbor sampling (Liu et al., WWW 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNBaseline(nn.Module):
    """2-layer GCN baseline for node classification."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 2)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h)


class SemiGNN(nn.Module):
    """Semi-supervised graph attentive network (Wang et al., ICDM 2019).

    Multi-view attention over neighbors + labeled-neighbor regularizer.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        raise NotImplementedError("hierarchical attention layers + semi-supervised loss")

    def forward(self, x, edge_index):
        raise NotImplementedError


def build_model(name: str, in_dim: int, **kwargs) -> nn.Module:
    models = {"gcn": GCNBaseline, "semignn": SemiGNN}
    if name not in models:
        raise ValueError(f"unknown model: {name!r} (choices: {sorted(models)})")
    return models[name](in_dim, **kwargs)
