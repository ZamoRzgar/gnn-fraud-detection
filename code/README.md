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
# main model (sampler-free) with a tuned filter keep-ratio:
python -m src.train --dataset yelp --model ours_v2 --rho 0.5 --epochs 100
python -m src.train --dataset yelp --model ours_v2 --loss focal --gamma 2.0
# ablation variants of the proposed model:
python -m src.train --dataset yelp --model ours_nofilter --epochs 100
python -m src.train --dataset yelp --model ours_nosampler --epochs 100
```

Models: `gcn` (2-layer GCN on the flattened homo graph), `semignn`
(multi-view hierarchical attention + semi-supervised graph loss; aux loss
weight via `--aux_weight`, 0 disables), `ours_v2` (the main model:
imbalance- and camouflage-aware multi-relation GNN; CARE-GNN-style
similarity filtering with keep-ratio `--rho`, attention aggregation and
relation fusion), the original `ours` (adds PC-GNN-style balanced sampling),
and the ablations `ours_nofilter` (similarity filtering removed) and
`ours_nosampler` (fraud-bias of the sampler removed, uniform sampling
within the same degree cap). The classification loss is CE by default or
focal loss with `--loss focal --gamma G`; class weighting can be disabled
with `--no_cw`. Every run appends its config + metrics to
`results/results.csv`; summarize with `python -m src.table [--meanstd|--all]`.

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
