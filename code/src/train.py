"""Training entry point.

Example:
    python -m src.train --dataset yelp --model gcn --epochs 100
    python -m src.train --dataset amazon --model ours --epochs 100 --aux_weight 0
"""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data import load_dataset
from src.evaluate import auroc, auprc, f1_macro, recall_at_k
from src.models import build_model

RESULTS_CSV = Path(__file__).resolve().parent.parent / "results" / "results.csv"


def parse_args():
    p = argparse.ArgumentParser(description="GNN fraud detection training")
    p.add_argument("--dataset", choices=["yelp", "amazon"], required=True)
    p.add_argument(
        "--model",
        choices=["gcn", "semignn", "ours", "ours_nofilter", "ours_nosampler"],
        default="gcn",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--aux_weight",
        type=float,
        default=0.1,
        help="weight of the model's aux_loss (0 disables it)",
    )
    p.add_argument(
        "--no_cw",
        action="store_true",
        help="disable class-weighted cross-entropy; with a sampler model this "
        "makes balanced sampling the only imbalance mechanism",
    )
    return p.parse_args()


def model_forward(model, data):
    """Model-agnostic forward call: dispatch on the model's interface flags."""
    kwargs = {}
    if getattr(model, "uses_relations", False):
        kwargs["relation_edge_index"] = data.relation_edge_index
    else:
        kwargs["edge_index"] = data.edge_index
    if getattr(model, "needs_labels", False):
        kwargs["y"] = data.y
        kwargs["train_mask"] = data.train_mask
    return model(data.x, **kwargs)


def evaluate_split(model, data, mask):
    model.eval()
    with torch.no_grad():
        logits = model_forward(model, data)
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
    # counter the strong imbalance toward the benign class. --no_cw disables
    # this so balanced sampling (if the model has it) is the only mechanism.
    y_train = data.y[data.train_mask]
    counts = torch.bincount(y_train, minlength=2).float()
    class_weight = None if args.no_cw else counts.sum() / (2.0 * counts)
    print(f"train class counts: {counts.tolist()}, class_weight: {class_weight}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_auroc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model_forward(model, data)
        loss = F.cross_entropy(
            logits[data.train_mask], y_train, weight=class_weight
        )
        if args.aux_weight > 0 and hasattr(model, "aux_loss"):
            loss = loss + args.aux_weight * model.aux_loss(data)
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
    save_result(args, best_val_auroc, test)


def save_result(args, best_val_auroc, test):
    """Append this run's config + metrics to results/results.csv."""
    RESULTS_CSV.parent.mkdir(exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "model": args.model,
        "epochs": args.epochs,
        "hidden": args.hidden,
        "lr": args.lr,
        "seed": args.seed,
        "aux_weight": args.aux_weight,
        "class_weight": 0 if args.no_cw else 1,
        "best_val_auroc": round(best_val_auroc, 4),
        "test_auroc": round(test["auroc"], 4),
        "test_auprc": round(test["auprc"], 4),
        "test_f1_macro": round(test["f1_macro"], 4),
        "test_recall@100": round(test["recall@100"], 4),
    }
    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"saved -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
