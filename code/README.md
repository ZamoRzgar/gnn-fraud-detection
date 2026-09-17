# Code — GNN Fraud Detection

PyTorch + PyTorch Geometric implementation of the baselines and our
imbalance-aware approach, evaluated on YelpChi and Amazon fraud datasets.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The YelpChi/Amazon fraud benchmarks are distributed with CARE-GNN:
https://github.com/YingtongDou/CARE-GNN (see its README for download).
Place the `.mat` files under `data/` (gitignored). See `src/data.py`.

## Usage (once implemented)

```bash
python -m src.train --dataset yelp --model gcn --epochs 100
python -m src.train --dataset amazon --model semignn --epochs 100
```

## Layout

- `src/data.py` — dataset loading (YelpChi, Amazon)
- `src/models.py` — GCNBaseline, SemiGNN, (+ CARE-GNN / PC-GNN style model)
- `src/train.py` — training loop (argparse entry point)
- `src/evaluate.py` — AUROC, AUPRC, F1-macro, Recall@k
