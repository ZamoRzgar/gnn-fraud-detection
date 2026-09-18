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

## Usage

```bash
python -m src.train --dataset yelp --model gcn --epochs 100
python -m src.train --dataset amazon --model semignn --epochs 100
python -m src.train --dataset yelp --model ours --epochs 100
```

Models: `gcn` (2-layer GCN on the flattened homo graph), `semignn`
(multi-view hierarchical attention + semi-supervised graph loss; aux loss
weight via `--aux_weight`, 0 disables), `ours` (imbalance- and
camouflage-aware multi-relation GNN; CARE-GNN-style similarity filtering +
PC-GNN-style balanced sampling).

`verify_setup.py` runs a one-pass shape/gradient smoke test for all models
on both datasets (not a training run).

## Layout

- `src/data.py` — dataset loading (YelpChi, Amazon); returns PyG `Data` with
  the union graph (`edge_index`) and per-relation graphs
  (`relation_edge_index`)
- `src/models.py` — GCNBaseline, SemiGNN, ImbalanceAwareGNN (`build_model`)
- `src/train.py` — training loop (argparse entry point; class-weighted CE,
  optional aux loss, best-val checkpointing)
- `src/evaluate.py` — AUROC, AUPRC, F1-macro, Recall@k
