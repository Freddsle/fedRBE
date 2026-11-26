"""Shared scikit-learn scores for sample-aligned hard labels, one function per metric."""

import numpy as np
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    f1_score,
    matthews_corrcoef,
    multilabel_confusion_matrix,
    normalized_mutual_info_score,
)


def mcc(truth, labels, *, sample_weight=None) -> float:
    """Matthews correlation coefficient (multiclass generalization for more than two classes)."""
    return float(matthews_corrcoef(truth, labels, sample_weight=sample_weight))


def weighted_f1(truth, labels, *, sample_weight=None) -> float:
    """Per-class F1 averaged with the classes' true sample counts as weights."""
    return float(f1_score(truth, labels, sample_weight=sample_weight,
                          average="weighted", zero_division=0))


def macro_f1(truth, labels, *, sample_weight=None) -> float:
    """Per-class F1 averaged with equal class weights."""
    return float(f1_score(truth, labels, sample_weight=sample_weight,
                          average="macro", zero_division=0))


# Name in the result files -> metric
CLASSIFICATION_METRICS = {
    "MCC": mcc,
    "weighted_F1_score": weighted_f1,
    "macro_F1_score": macro_f1,
}


def classification_scores(truth, labels, *, sample_weight=None) -> dict[str, float]:
    """Every metric of CLASSIFICATION_METRICS; predicted labels must use the true class meanings."""
    return {name: metric(truth, labels, sample_weight=sample_weight)
            for name, metric in CLASSIFICATION_METRICS.items()}


def confusion_counts(truth, labels, classes) -> dict[str, int]:
    """
    One-vs-rest TP, TN, FP and FN of every class in *classes*, keyed "TP|<class>".
    Every true and predicted label must be one of *classes*.
    """
    unknown = (set(np.unique(truth)) | set(np.unique(labels))) - set(classes)
    if unknown:
        raise ValueError(f"Labels {sorted(map(str, unknown))} are not among the classes "
                         f"{[str(cls) for cls in classes]}")
    counts = {}
    matrices = multilabel_confusion_matrix(truth, labels, labels=classes)
    for cls, ((tn, fp), (fn, tp)) in zip(classes, matrices):
        counts.update({f"TP|{cls}": int(tp), f"TN|{cls}": int(tn),
                       f"FP|{cls}": int(fp), f"FN|{cls}": int(fn)})
    return counts


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
