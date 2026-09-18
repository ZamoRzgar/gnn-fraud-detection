"""Training entry point.

Example:
    python -m src.train --dataset yelp --model gcn --epochs 100
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from src.data import load_dataset
from src.evaluate import auroc, auprc, f1_macro, recall_at_k
from src.models import build_model


def parse_args():
    p = argparse.ArgumentParser(description="GNN fraud detection training")
    p.add_argument("--dataset", choices=["yelp", "amazon"], required=True)
    p.add_argument("--model", choices=["gcn", "semignn"], default="gcn")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def evaluate_split(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    score = logits.softmax(dim=1)[:, 1].cpu().numpy()
    y_true = data.y[mask].cpu().numpy()
    y_pred = logits.argmax(dim=1)[mask].cpu().numpy()
    return {
        "auroc": auroc(y_true, score[mask.cpu().numpy()]),
        "auprc": auprc(y_true, score[mask.cpu().numpy()]),
        "f1_macro": f1_macro(y_true, y_pred),
        "recall@100": recall_at_k(y_true, score[mask.cpu().numpy()], 100),
    }


def main():
    args = parse_args()
    print(f"config: {vars(args)}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = load_dataset(args.dataset)
    model = build_model(args.model, in_dim=data.num_features, hidden_dim=args.hidden)

    # Inverse-class-frequency weights (computed on the training split) to
    # counter the strong imbalance toward the benign class.
    y_train = data.y[data.train_mask]
    counts = torch.bincount(y_train, minlength=2).float()
    class_weight = counts.sum() / (2.0 * counts)
    print(f"train class counts: {counts.tolist()}, weights: {class_weight.tolist()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_auroc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(
            logits[data.train_mask], y_train, weight=class_weight
        )
        loss.backward()
        optimizer.step()

        val = evaluate_split(model, data, data.val_mask)
        if val["auroc"] > best_val_auroc:
            best_val_auroc = val["auroc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch:3d} | loss {loss.item():.4f} "
                f"| val auroc {val['auroc']:.4f} auprc {val['auprc']:.4f}"
            )

    model.load_state_dict(best_state)
    test = evaluate_split(model, data, data.test_mask)
    print(f"best val auroc: {best_val_auroc:.4f}")
    print(
        f"test | auroc {test['auroc']:.4f} | auprc {test['auprc']:.4f} "
        f"| f1_macro {test['f1_macro']:.4f} | recall@100 {test['recall@100']:.4f}"
    )


if __name__ == "__main__":
    main()
