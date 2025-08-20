"""
Metrics for evaluating clustering models.
"""

# Author: Atif Khurshid
# Created: 2025-08-19
# Modified: 2025-08-20
# Version: 1.1
# Changelog:
#     - 2025-08-20: Removed adjusted maximum cluster assignment score

import numpy as np
from munkres import Munkres
from scipy.special import comb
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import contingency_matrix


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


def maximum_cluster_assignment_score(
        labels_true: list | np.ndarray,
        labels_pred: list | np.ndarray
    ) -> float:
    """
    Compute the maximum cluster assignment (MCA) index.
    Adapted from Kraus and Kestler BMC Bioinformatics 2010, 11:169
    http://www.biomedcentral.com/1471-2105/11/169

    Parameters
    ----------
    labels_true : list | np.ndarray
        True labels of the data.
    labels_pred : list | np.ndarray
        Predicted labels by the clustering model.

    Returns
    -------
    float
        Maximum cluster assignment index.
    """
    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)

    similarity_matrix = np.zeros((len(classes_true), len(classes_pred)))
    for i, true_class in enumerate(classes_true):
        for j, pred_class in enumerate(classes_pred):
            similarity_matrix[i, j] = np.sum((labels_true == true_class) & (labels_pred == pred_class))

    true_ind, pred_ind = linear_sum_assignment(similarity_matrix, maximize=True)
    mca = np.sum(similarity_matrix[true_ind, pred_ind]) / np.sum(similarity_matrix)

    return mca


def intra_cluster_variability(
        features: np.ndarray,
        labels: np.ndarray,
        metric: str = 'euclidean'
    ):
    """
    Calculate the average intra-cluster variability (mean pairwise distance within each cluster).

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    labels : np.ndarray
        Cluster labels of shape (n_samples,).
    metric : str, default='euclidean'
        Distance metric for pairwise_distances.
    

    Returns
    -------
    np.ndarray
        Mean intra-cluster variability across all clusters.
    """
    dists = pairwise_distances(features, metric=metric)
    unique_labels = np.unique(labels)
    variances = []
    for label in unique_labels:
        idx = np.where(labels == label)[0]
        if len(idx) < 2:
            variances.append(0.0)
            continue
        sub_dists = dists[np.ix_(idx, idx)]
        # Only upper triangle, excluding diagonal
        triu = sub_dists[np.triu_indices_from(sub_dists, k=1)]
        if len(triu) > 0:
            variances.append(triu.mean())

    return np.array(variances)


def inter_cluster_variability(
        features: np.ndarray,
        labels: np.ndarray,
        metric: str = 'euclidean'
    ):
    """
    Calculate the average inter-cluster variability (mean pairwise distance between clusters).

    Parameters
    ----------
    features : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    labels : np.ndarray
        Cluster labels of shape (n_samples,).
    metric : str, default='euclidean'
        Distance metric for pairwise_distances.

    Returns
    -------
    float
        Mean inter-cluster variability across all cluster pairs.
    """
    unique_labels = np.unique(labels)
    centroids = np.array([features[labels == label].mean(axis=0) for label in unique_labels])
    centroid_dists = pairwise_distances(centroids, metric=metric)
    variances = []
    for i in range(len(unique_labels)):
        # Only consider distances to other clusters (exclude diagonal)
        other_dists = np.delete(centroid_dists[i], i)
        if len(other_dists) > 0:
            variances.append(other_dists.mean())
        else:
            variances.append(0.0)

    return np.array(variances)
