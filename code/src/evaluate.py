"""Evaluation metrics standard for graph fraud detection
(CARE-GNN / PC-GNN benchmarks): AUROC, AUPRC, F1-macro, Recall@k.
"""

import numpy as np


def auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    raise NotImplementedError("sklearn.metrics.roc_auc_score")


def auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    raise NotImplementedError("sklearn.metrics.average_precision_score")


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    raise NotImplementedError("sklearn.metrics.f1_score(average='macro')")


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """Recall among the top-k highest-scoring (most suspicious) nodes."""
    raise NotImplementedError
