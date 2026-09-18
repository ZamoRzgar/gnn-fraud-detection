"""Shape/gradient smoke test: loads both datasets, builds all three models,
runs one forward + backward pass each, and prints per-relation edge counts.

NOT a training run — one optimizer-free backward pass per model.

Usage: python verify_setup.py
"""

import torch
import torch.nn.functional as F

from src.data import load_dataset
from src.models import build_model
from src.train import model_forward

for dataset in ["yelp", "amazon"]:
    data = load_dataset(dataset)
    print(f"\n=== {dataset}: nodes={data.num_nodes} feats={data.num_features} "
          f"homo_edges={data.edge_index.shape[1]} ===")
    for rel, ei in sorted(data.relation_edge_index.items()):
        print(f"  relation {rel}: {ei.shape[1]} edges")

    for model_name in ["gcn", "semignn", "ours"]:
        torch.manual_seed(0)
        model = build_model(model_name, in_dim=data.num_features)
        model.train()
        model.zero_grad()
        logits = model_forward(model, data)
        assert logits.shape == (data.num_nodes, 2), f"bad shape {logits.shape}"

        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        if hasattr(model, "aux_loss"):
            loss = loss + 0.1 * model.aux_loss(data)
        loss.backward()

        n_with_grad = sum(
            1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        n_params = sum(1 for _ in model.parameters())
        print(
            f"  {model_name:8s} logits {tuple(logits.shape)} "
            f"loss {loss.item():.4f} | params with nonzero grad: {n_with_grad}/{n_params}"
        )

print("\nOK: all models forward/backward on both datasets.")
