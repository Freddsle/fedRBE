"""Shared scikit-learn scores for sample-aligned hard labels."""

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    f1_score,
    matthews_corrcoef,
    normalized_mutual_info_score,
)


def classification_scores(truth, labels, *, sample_weight=None) -> dict[str, float]:
    """MCC and support-weighted F1; predicted labels must use the true class meanings."""
    return {
        "MCC": float(matthews_corrcoef(truth, labels, sample_weight=sample_weight)),
        "F1_score": float(f1_score(
            truth, labels, sample_weight=sample_weight, average="weighted", zero_division=0,
        )),
    }


def classification_scores_from_confusion(matrix) -> dict[str, float]:
    """Score counts with true rows and predicted columns in the same class order."""
    counts = np.asarray(matrix)
    if (counts.ndim != 2 or counts.shape[0] != counts.shape[1] or not counts.size
            or not np.isfinite(counts).all() or (counts < 0).any() or counts.sum() <= 0):
        raise ValueError("Confusion matrix must be square, finite, nonnegative with positive total")
    truth, labels = np.nonzero(counts)
    # Score cells as weighted samples without expanding counts into observations.
    return classification_scores(truth, labels, sample_weight=counts[truth, labels])


def clustering_scores(truth, labels) -> dict[str, float]:
    """ARI and arithmetic AMI/NMI between partitions, independent of label names."""
    return {
        "ARI": float(adjusted_rand_score(truth, labels)),
        "AMI": float(adjusted_mutual_info_score(truth, labels, average_method="arithmetic")),
        "NMI": float(normalized_mutual_info_score(truth, labels, average_method="arithmetic")),
    }
