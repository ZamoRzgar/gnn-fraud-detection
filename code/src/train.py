"""Training entry point.

Example:
    python -m src.train --dataset yelp --model gcn --epochs 100
"""

import argparse


def parse_args():
    p = argparse.ArgumentParser(description="GNN fraud detection training")
    p.add_argument("--dataset", choices=["yelp", "amazon"], required=True)
    p.add_argument("--model", choices=["gcn", "semignn"], default="gcn")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"config: {vars(args)}")
    # TODO: data = load_dataset(args.dataset)
    # TODO: model = build_model(args.model, in_dim=data.num_features)
    # TODO: train loop with class-weighted loss; eval via src.evaluate


if __name__ == "__main__":
    main()
