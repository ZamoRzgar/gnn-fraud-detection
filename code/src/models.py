"""Model definitions.

Baselines and target models from the surveyed papers:
- GCN: standard baseline (Kipf & Welling 2017)
- SemiGNN: Wang et al., ICDM 2019 (arXiv:2003.01171)
- Planned: CARE-GNN-style camouflage-resistant aggregation (Dou et al., CIKM 2020)
  and PC-GNN-style balanced neighbor sampling (Liu et al., WWW 2021)
"""

import torch
import torch.nn as nn


class GCNBaseline(nn.Module):
    """2-layer GCN baseline for node classification."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        raise NotImplementedError("GCN encoder + linear classifier head")

    def forward(self, x, edge_index):
        raise NotImplementedError


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
