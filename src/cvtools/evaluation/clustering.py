"""
Evaluation functions for clustering models.
"""

# Author: Atif Khurshid
# Created: 2025-06-22
# Modified: 2025-08-19
# Version: 2.0
# Changelog:
#     - 2025-08-01: Added documentation and type hints.
#     - 2025-08-04: Scaled evaluation metrics.
#     - 2025-08-14: Added Silhouette Score.
#     - 2025-08-14: Added Accuracy, F1 Score, and Clustering Purity.
#     - 2025-08-19: Added Maximum Cluster Assignment Index.
#     - 2025-08-19: Refactored clustering evaluation functions.

from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import adjusted_mutual_info_score

from .metrics import clustering_accuracy
from .metrics import clustering_f_measure
from .metrics import clustering_purity
from .metrics import adjusted_maximum_cluster_assignment_score


def evaluate_clustering(
        features: list | np.ndarray,
        labels_true: list | np.ndarray,
        labels_pred: list | np.ndarray,
        metric: str = "euclidean",
        report: bool = True,
    ) -> dict | None:
    """
    Evaluate clustering model performance.
    Computes and optionally prints the Adjusted Rand Index, Normalized Mutual Information,
    and Fowlkes-Mallows Score.

    Parameters
    -----------
    features : list or np.ndarray
        Feature vectors of the data.
    labels_true : list or np.ndarray
        True labels of the data.
    labels_pred : list or np.ndarray
        Predicted labels by the clustering model.
    metric: str, optional
        The distance metric to use for the silhouette score. Default is "euclidean".
    report : bool, optional
        If True, prints the evaluation scores. If False, returns them as a dictionary.

    Returns
    --------
    dict or None
        If report is False, returns a dictionary with evaluation scores.
    
    Examples
    ---------
    >>> features = np.array([[1, 2], [2, 3], [3, 4], [5, 6]])
    >>> labels_true = [0, 1, 1, 0, 2, 2]
    >>> labels_pred = [0, 0, 1, 1, 2, 2]
    >>> evaluate_clustering(features, labels_true, labels_pred, report=True)
    Clustering Accuracy: 66.67
    F1 Score: 66.67
    Clustering Purity: 66.67
    Adjusted Rand Index: 66.7
    Normalized Mutual Information: 57.74
    Fowlkes-Mallows Score: 66.67
    Silhouette Score: 80.0
    >>> scores = evaluate_clustering(features, labels_true, labels_pred, report=False)
    >>> print(scores)
    {
        "Clustering Accuracy": 66.67,
        "F1 Score": 66.67,
        "Clustering Purity": 66.67,
        "Adjusted Rand Index": 66.7,
        "Normalized Mutual Information": 57.74,
        "Fowlkes-Mallows Score": 66.67,
        "Silhouette Score": 80.0
    }
    """

    acc = clustering_accuracy(labels_true, labels_pred) * 100
    f1s = clustering_f_measure(labels_true, labels_pred) * 100
    pur = clustering_purity(labels_true, labels_pred) * 100
    mca = adjusted_maximum_cluster_assignment_score(labels_true, labels_pred) * 100
    ari = adjusted_rand_score(labels_true, labels_pred) * 100
    nmi = adjusted_mutual_info_score(labels_true, labels_pred) * 100
    fms = fowlkes_mallows_score(labels_true, labels_pred) * 100
    sil = silhouette_score(features, labels_pred, metric=metric) * 100

    scores = {
        "Clustering Accuracy": float(acc),
        "F1 Score": float(f1s),
        "Clustering Purity": float(pur),
        "Adjusted MCA Index": float(mca),
        "Adjusted Rand Index": float(ari),
        "Normalized Mutual Information": float(nmi),
        "Fowlkes-Mallows Score": float(fms),
        "Silhouette Score": float(sil),
    }

    if report:
        pprint(scores)
    else:
        return scores
