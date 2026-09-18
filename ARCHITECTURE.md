# Architecture

## Pipeline

```
.mat files (CARE-GNN release)
      │
      ▼
data.py        loadmat → PyG Data (x, edge_index, y, train/val/test masks)
      │
      ▼
models.py      build_model(name, in_dim) → nn.Module (2 logits per node)
      │
      ▼
train.py       full-batch training, class-weighted CE loss,
               best-val checkpointing by AUROC
      │
      ▼
evaluate.py    AUROC, AUPRC, F1-macro, Recall@k on the test mask
```

Everything runs full-batch on CPU: YelpChi (~45k nodes) and Amazon (~12k
nodes) are small enough that mini-batching is unnecessary for baselines.

## Module responsibilities (`code/src/`)

- `data.py` — downloads nothing itself; expects the `.mat` files under
  `code/data/`. Loads features/labels/adjacency with `scipy.io.loadmat`,
  builds a `torch_geometric.data.Data` object, and creates the
  train/val/test split with a fixed seed. Exposes `load_yelp()`,
  `load_amazon()`, and `load_dataset(name)`.
- `models.py` — model zoo. `GCNBaseline` (2-layer GCN, hidden 64, ReLU,
  dropout 0.5, linear head → 2 logits); `SemiGNN` (stub for now); the
  proposed model will be added here later. `build_model(name, in_dim)` is
  the single construction entry point.
- `train.py` — argparse entry point (`python -m src.train --dataset …
  --model …`). Sets seeds, loads the dataset, builds the model, runs the
  training loop: `CrossEntropyLoss` with inverse-class-frequency weights,
  Adam, per-epoch validation by AUROC, keeps the best-val state dict, and
  evaluates it on the test mask at the end.
- `evaluate.py` — thin wrappers over sklearn: `auroc`, `auprc`,
  `f1_macro`, `recall_at_k`. All take numpy arrays and return floats.

## Data flow

Each dataset is a single homogeneous `torch_geometric.data.Data`:

- `x` — dense float tensor `[num_nodes, num_features]`.
- `edge_index` — int64 `[2, num_edges]`, built from the combined
  ("homo") adjacency, symmetrized, self-loops removed.
- `y` — int64 `[num_nodes]`; 1 = fraud (fake review / fraudulent
  reviewer), 0 = benign.
- `train_mask` / `val_mask` / `test_mask` — boolean `[num_nodes]`,
  random 40/20/40 split (CARE-GNN convention), seeded for reproducibility.

The source data is multi-relation (YelpChi: R-U-R, R-S-R, R-T-R; Amazon:
U-P-U, U-S-U, U-V-U). The `homo` adjacency distributed with CARE-GNN is
the union of all relations, which is what the homogeneous baseline uses.

## Design decisions

- **PyTorch + PyTorch Geometric**: standard stack for the surveyed papers;
  PyG's `GCNConv` handles normalization and message passing.
- **Homogeneous view first**: all baseline papers (CARE-GNN, PC-GNN) report
  plain-GCN numbers on the flattened `homo` graph, so our baseline is
  directly comparable. Relation-aware modeling is deferred to the proposed
  model phase; `RELATIONS` in `data.py` already records the relation names.
- **Class imbalance via loss weighting**: fraud is the minority class, so
  training uses `CrossEntropyLoss(weight=)` with weights inversely
  proportional to class frequency on the training split. Sampling-based
  balancing (PC-GNN style) comes later in the proposed model.
- **Evaluation**: thresholds matter less than ranking for fraud triage, so
  AUROC/AUPRC are primary; F1-macro uses the argmax prediction; Recall@k
  mimics an investigator reviewing the top-k most suspicious nodes.
- **Extensibility**: new models only need a class in `models.py` and an
  entry in `build_model()`'s registry; `train.py` and `evaluate.py` are
  model-agnostic (any `nn.Module` mapping `(x, edge_index) → logits`).
