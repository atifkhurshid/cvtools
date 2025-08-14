"""
Evaluation functions for clustering models.
"""

# Author: Atif Khurshid
# Created: 2025-06-22
# Modified: 2025-08-14
# Version: 1.3
# Changelog:
#     - 2025-08-01: Added documentation and type hints.
#     - 2025-08-04: Scaled evaluation metrics.
#     - 2025-08-14: Added Silhouette Score.
#     - 2025-08-14: Added Accuracy, F1 Score, and Clustering Purity.

from pprint import pprint

import numpy as np
from munkres import Munkres
from scipy.special import comb
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix


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
    ari = adjusted_rand_score(labels_true, labels_pred) * 100
    nmi = adjusted_mutual_info_score(labels_true, labels_pred) * 100
    fms = fowlkes_mallows_score(labels_true, labels_pred) * 100
    sil = silhouette_score(features, labels_pred, metric=metric) * 100

    scores = {
        "Clustering Accuracy": float(acc),
        "F1 Score": float(f1s),
        "Clustering Purity": float(pur),
        "Adjusted Rand Index": float(ari),
        "Normalized Mutual Information": float(nmi),
        "Fowlkes-Mallows Score": float(fms),
        "Silhouette Score": float(sil),
    }

    if report:
        pprint(scores)
    else:
        return scores



def clustering_accuracy(
        labels_true: list | np.ndarray,
        labels_pred: list | np.ndarray
    ) -> float:
    """
    Calculate clustering accuracy using the Kuhn-Munkres algorithm.
    Adapted from https://github.com/hexiangnan/CoNMF

    Parameters
    ----------
    labels_true : list | np.ndarray
        True labels of the data.
    labels_pred : list | np.ndarray
        Predicted labels by the clustering model.

    Returns
    -------
    float
        Clustering accuracy score.
    
    Examples
    ---------
    >>> labels_true = [0, 1, 1, 0, 2, 2]
    >>> labels_pred = [0, 0, 1, 1, 2, 2]
    >>> clustering_accuracy(labels_true, labels_pred)
    0.6666666666666666
    """
    n_samples = labels_true.shape[0]
    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
            or classes.shape[0] == clusters.shape[0] == len(labels_true)):
        return 1.0
    
    contingency = contingency_matrix(labels_true, labels_pred)
    # Type: <type 'numpy.ndarray'>:rows are clusters, cols are classes
    contingency = -contingency

    contingency = contingency.tolist()
    m = Munkres() # Best mapping by using Kuhn-Munkres algorithm
    map_pairs = m.compute(contingency) # best match to find the minimum cost
    sum_value = 0
    for key,value in map_pairs:
        sum_value = sum_value + contingency[key][value]
    
    return float(-sum_value) / n_samples


def clustering_f_measure(
        labels_true: list | np.ndarray,
        labels_pred: list | np.ndarray
    ) -> float:
    """
    Compute the F-measure for clustering results.
    Adapted from https://github.com/hexiangnan/CoNMF

    Parameters
    ----------
    labels_true : list | np.ndarray
        True labels of the data.
    labels_pred : list | np.ndarray
        Predicted labels by the clustering model.

    Returns
    -------
    float
        F-measure score.
    """
    def comb2(n):
        # the exact version is faster for k == 2: use it by default globally in
        # this module instead of the float approximate variant
        return comb(n, 2, exact=True)

    classes = np.unique(labels_true)
    clusters = np.unique(labels_pred)
    # Special limit cases: no clustering since the data is not split;
    # or trivial clustering where each document is assigned a unique cluster.
    # These are perfect matches hence return 1.0.
    if (classes.shape[0] == clusters.shape[0] == 1
            or classes.shape[0] == clusters.shape[0] == 0
            or classes.shape[0] == clusters.shape[0] == len(labels_true)):
        return 1.0

    contingency = contingency_matrix(labels_true, labels_pred)

    # Compute the ARI using the contingency data
    TP_plus_FP = sum(comb2(n_c) for n_c in contingency.sum(axis=1)) # TP+FP
    
    TP_plus_FN = sum(comb2(n_k) for n_k in contingency.sum(axis=0)) # TP+FN
    
    TP = sum(comb2(n_ij) for n_ij in contingency.flatten()) # TP
    
    P = float(TP) / TP_plus_FP
    R = float(TP) / TP_plus_FN
    
    return 2*P*R/(P+R)


def clustering_purity(
        labels_true: list | np.ndarray,
        labels_pred: list | np.ndarray
    ) -> float:
    """
    Compute the clustering purity.
    Adapted from https://stackoverflow.com/a/51672699

    Parameters
    ----------
    labels_true : list | np.ndarray
        True labels of the data.
    labels_pred : list | np.ndarray
        Predicted labels by the clustering model.

    Returns
    -------
    float
        Clustering purity score.
    """
    contingency = contingency_matrix(labels_true, labels_pred)

    purity = np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)

    return purity
