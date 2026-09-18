"""Evaluation metrics standard for graph fraud detection
(CARE-GNN / PC-GNN benchmarks): AUROC, AUPRC, F1-macro, Recall@k.
"""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Recall among the top-k highest-scoring (most suspicious) nodes."""
    k = min(k, len(y_true))
    top_k = np.argsort(y_score)[::-1][:k]
    positives = y_true.sum()
    if positives == 0:
        return 0.0
    return float(y_true[top_k].sum() / positives)
