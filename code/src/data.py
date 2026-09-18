"""Dataset loaders for the YelpChi and Amazon fraud benchmarks.

The datasets (heterogeneous multi-relation graphs, stored as .mat files) are
distributed with CARE-GNN: https://github.com/YingtongDou/CARE-GNN
The .mat files (YelpChi.mat, Amazon.mat) live under ../data/ (gitignored).

Each .mat holds: 'features' (sparse CSR node features), 'label' (1 = fraud),
relation adjacency matrices (yelp: net_rur/net_rsr/net_rtr; amazon:
net_upu/net_usu/net_uvu), and 'homo', the union of all relations.

The returned Data object carries both the union graph ('edge_index', from
'homo') and the per-relation graphs ('relation_edge_index', a dict mapping
relation name -> [2, E] long tensor). Relation matrices are symmetrized and
stripped of self-loops; in practice they ship symmetric with zero diagonal.
Models that only need the flat graph (GCNBaseline) use 'edge_index';
multi-relation models (SemiGNN, ImbalanceAwareGNN) use 'relation_edge_index'.

Split convention (following CARE-GNN's train.py): stratified random split
with random_state=2. CARE-GNN itself uses 40% train / 60% test with no
validation set; we carve the 60% remainder into val 20% / test 40% so the
training loop can do best-val model selection. For Amazon, nodes 0-3304 are
unlabeled (CARE-GNN excludes them from the split via labels[3305:]); they
stay in the graph for message passing but appear in no mask.

Features are L2 row-normalized, as in CARE-GNN (`sklearn.preprocessing
.normalize`).
"""

from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from torch_geometric.data import Data

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

RELATIONS = {
    "yelp": ["rur", "rsr", "rtr"],
    "amazon": ["upu", "usu", "uvu"],
}

# Amazon nodes 0..3304 are unlabeled (CARE-GNN convention).
UNLABELED_PREFIX = {"yelp": 0, "amazon": 3305}


def _load_mat(key: str, filename: str) -> Data:
    mat_path = DATA_DIR / filename
    if not mat_path.exists():
        raise FileNotFoundError(
            f"{mat_path} not found. Download it from the CARE-GNN repo "
            "(see code/README.md) and place it under code/data/."
        )
    mat = sio.loadmat(mat_path)

    x = torch.from_numpy(normalize(mat["features"].todense().A)).float()
    y = torch.from_numpy(mat["label"].flatten()).long()

    # 'homo' is the union of all relations; already symmetric, no self-loops.
    homo = mat["homo"].tocoo()
    edge_index = torch.from_numpy(np.vstack([homo.row, homo.col])).long()

    # Per-relation graphs for multi-relation models. Symmetrize and drop
    # self-loops defensively (the shipped matrices are already symmetric
    # with zero diagonal).
    relation_edge_index = {}
    for rel in RELATIONS[key]:
        adj = mat[f"net_{rel}"].tocsr()
        adj = (adj + adj.T).tocsr()
        adj.setdiag(0)
        adj.eliminate_zeros()
        coo = adj.tocoo()
        relation_edge_index[rel] = torch.from_numpy(
            np.vstack([coo.row, coo.col])
        ).long()

    num_nodes = x.shape[0]
    labeled_idx = np.arange(UNLABELED_PREFIX[key], num_nodes)
    labeled_y = y[labeled_idx].numpy()

    # Stratified 40% train / 60% rest (CARE-GNN), then rest -> 20% val / 40% test.
    idx_train, idx_rest = train_test_split(
        labeled_idx, stratify=labeled_y, test_size=0.60, random_state=2, shuffle=True
    )
    idx_val, idx_test = train_test_split(
        idx_rest,
        stratify=y[idx_rest].numpy(),
        test_size=2 / 3,
        random_state=2,
        shuffle=True,
    )

    masks = {}
    for split, idx in [("train", idx_train), ("val", idx_val), ("test", idx_test)]:
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[torch.from_numpy(np.asarray(idx))] = True
        masks[f"{split}_mask"] = mask

    return Data(
        x=x,
        edge_index=edge_index,
        y=y,
        relation_edge_index=relation_edge_index,
        **masks,
    )


def load_yelp() -> Data:
    """Load YelpChi (review spam). Labels: 1 = fake review, 0 = genuine."""
    return _load_mat("yelp", "YelpChi.mat")


def load_amazon() -> Data:
    """Load Amazon fraud dataset. Labels: 1 = fraudulent reviewer."""
    return _load_mat("amazon", "Amazon.mat")


def load_dataset(name: str):
    if name == "yelp":
        return load_yelp()
    if name == "amazon":
        return load_amazon()
    raise ValueError(f"unknown dataset: {name!r} (expected 'yelp' or 'amazon')")
