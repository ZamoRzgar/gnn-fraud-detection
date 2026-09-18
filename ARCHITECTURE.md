# Architecture

## Pipeline

```
.mat files (CARE-GNN release)
      │
      ▼
data.py        loadmat → PyG Data (x, edge_index, relation_edge_index,
               y, train/val/test masks)
      │
      ▼
models.py      build_model(name, in_dim) → nn.Module (2 logits per node)
      │
      ▼
train.py       full-batch training, class-weighted CE loss (+ optional
               model aux_loss), best-val checkpointing by AUROC
      │
      ▼
evaluate.py    AUROC, AUPRC, F1-macro, Recall@k on the test mask
```

Everything runs full-batch on CPU: YelpChi (~45k nodes) and Amazon (~12k
nodes) are small enough that mini-batching is unnecessary for baselines.
Edge-wise computations in the multi-relation models are chunked
(`EDGE_CHUNK` in `models.py`) so the ~8M-edge relations stay within a few
GB of RAM.

## Module responsibilities (`code/src/`)

- `data.py` — downloads nothing itself; expects the `.mat` files under
  `code/data/`. Loads features/labels/adjacency with `scipy.io.loadmat`,
  builds a `torch_geometric.data.Data` object, and creates the
  train/val/test split with a fixed seed. Exposes `load_yelp()`,
  `load_amazon()`, and `load_dataset(name)`.
- `models.py` — model zoo, plus shared helpers (chunked edge scoring,
  vectorized per-node top-k selection, dot-product attention aggregation):
  - `GCNBaseline` — 2-layer GCN on the flattened homo graph, hidden 64,
    ReLU, dropout 0.5, linear head → 2 logits.
  - `SemiGNN` (Wang et al., ICDM 2019) — per-relation node-level attention,
    per-view MLP, view-level attention, 2-class head; `aux_loss()` adds the
    paper's semi-supervised graph loss (DeepWalk-style negative sampling on
    edges incident to labeled nodes).
  - `ImbalanceAwareGNN` ("ours") — the proposed model. Per relation:
    (a) CARE-GNN-style similarity filtering (MLP scores center–neighbor
    similarity, top `keep_ratio` kept) and (b) PC-GNN-style balanced
    sampling (degree cap + fraud-favoring weighted sampling using training
    labels), then (c) attention aggregation within each relation and
    attention fusion across relations.
  - `build_model(name, in_dim)` is the single construction entry point
    (`"gcn"`, `"semignn"`, `"ours"`).
- `train.py` — argparse entry point (`python -m src.train --dataset …
  --model …`). Sets seeds, loads the dataset, builds the model, runs the
  training loop: `CrossEntropyLoss` with inverse-class-frequency weights
  (plus `aux_weight * model.aux_loss(data)` when the model exposes one and
  `--aux_weight > 0`), Adam, per-epoch validation by AUROC, keeps the
  best-val state dict, and evaluates it on the test mask at the end. The
  forward call is model-agnostic (`model_forward`): it inspects the model's
  `uses_relations` / `needs_labels` class flags to decide which graph and
  label tensors to pass.
- `evaluate.py` — thin wrappers over sklearn: `auroc`, `auprc`,
  `f1_macro`, `recall_at_k`. All take numpy arrays and return floats.

## Data flow

Each dataset is a single `torch_geometric.data.Data`:

- `x` — dense float tensor `[num_nodes, num_features]`.
- `edge_index` — int64 `[2, num_edges]`, built from the combined
  ("homo") adjacency (union of all relations), symmetrized, self-loops
  removed. Used by homogeneous models (GCNBaseline) and by SemiGNN's
  aux loss.
- `relation_edge_index` — dict mapping relation name → int64 `[2, E]`
  per-relation graphs (YelpChi: `rur`/`rsr`/`rtr`; Amazon: `upu`/`usu`/
  `uvu`), symmetrized, self-loops removed. Used by the multi-relation
  models (SemiGNN, ImbalanceAwareGNN).
- `y` — int64 `[num_nodes]`; 1 = fraud (fake review / fraudulent
  reviewer), 0 = benign.
- `train_mask` / `val_mask` / `test_mask` — boolean `[num_nodes]`,
  random 40/20/40 split (CARE-GNN convention), seeded for reproducibility.

## Design decisions

- **PyTorch + PyTorch Geometric**: standard stack for the surveyed papers;
  PyG's `GCNConv` handles normalization and message passing.
- **Homogeneous view for the GCN baseline**: all baseline papers (CARE-GNN,
  PC-GNN) report plain-GCN numbers on the flattened `homo` graph, so our
  baseline is directly comparable. The relation-aware models consume
  `relation_edge_index` instead.
- **Class imbalance, two mechanisms**: (1) loss weighting —
  `CrossEntropyLoss(weight=)` with weights inversely proportional to class
  frequency on the training split, used by all models; (2) sampling —
  ImbalanceAwareGNN additionally balances each node's aggregated
  neighborhood via fraud-favoring, degree-capped sampling (PC-GNN idea).
- **Camouflage**: fraudsters mimic benign nodes, breaking homophily.
  ImbalanceAwareGNN filters each node's neighbors by a learned similarity
  score (CARE-GNN idea, fixed keep-ratio instead of CARE-GNN's RL
  controller) so dissimilar neighbors are dropped before aggregation.
- **Evaluation**: thresholds matter less than ranking for fraud triage, so
  AUROC/AUPRC are primary; F1-macro uses the argmax prediction; Recall@k
  mimics an investigator reviewing the top-k most suspicious nodes.
- **Extensibility**: new models only need a class in `models.py` and an
  entry in `build_model()`'s registry. `train.py` and `evaluate.py` are
  model-agnostic via the interface flags (`uses_relations`, `needs_labels`)
  and the optional `aux_loss(data)` hook, so the proposed model (or its
  future variants, e.g. an RL-controlled filtering threshold) plugs in
  without touching the training loop.
