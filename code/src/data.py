"""Dataset loaders for the YelpChi and Amazon fraud benchmarks.

The datasets (heterogeneous multi-relation graphs, stored as .mat files) are
distributed with CARE-GNN: https://github.com/YingtongDou/CARE-GNN
Download them per that repo's instructions and place under ../data/.
"""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

RELATIONS = {
    "yelp": ["rur", "rsr", "rtr"],
    "amazon": ["upu", "usu", "uvu"],
}


def load_yelp():
    """Load YelpChi (review spam). Labels: 1 = fake review, 0 = genuine.

    Returns a PyG HeteroData (or Data per relation) with node features,
    edges, labels, and train/val/test masks.
    """
    raise NotImplementedError(
        "Download YelpChi from the CARE-GNN repo (see code/README.md), "
        "then load yelp_homo.mat / relation .mat files from data/."
    )


def load_amazon():
    """Load Amazon fraud dataset. Labels: 1 = fraudulent reviewer.

    Same return convention as load_yelp().
    """
    raise NotImplementedError(
        "Download Amazon from the CARE-GNN repo (see code/README.md), "
        "then load amazon_homo.mat / relation .mat files from data/."
    )


def load_dataset(name: str):
    if name == "yelp":
        return load_yelp()
    if name == "amazon":
        return load_amazon()
    raise ValueError(f"unknown dataset: {name!r} (expected 'yelp' or 'amazon')")
