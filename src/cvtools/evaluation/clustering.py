"""
Evaluation functions for clustering models.
"""

# Author: Atif Khurshid
# Created: 2025-06-22
# Modified: 2025-08-20
# Version: 2.1
# Changelog:
#     - 2025-08-01: Added documentation and type hints.
#     - 2025-08-04: Scaled evaluation metrics.
#     - 2025-08-14: Added Silhouette Score.
#     - 2025-08-14: Added Accuracy, F1 Score, and Clustering Purity.
#     - 2025-08-19: Added Maximum Cluster Assignment Index.
#     - 2025-08-19: Refactored clustering evaluation functions.
#     - 2025-08-20: Added cluster stability evaluation.

from pprint import pprint
from typing import Callable

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import pairwise_distances_argmin

from .metrics import clustering_accuracy
from .metrics import clustering_f_measure
from .metrics import clustering_purity
from .metrics import maximum_cluster_assignment_score
from .metrics import silhouette_score


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
    mca = maximum_cluster_assignment_score(labels_true, labels_pred) * 100
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


def evaluate_clustering_stability(
        features: np.ndarray,
        labels_true: np.ndarray,
        n_clusters_list: list[int],
        evaluation_fn: Callable = maximum_cluster_assignment_score,
        n_repetitions: int = 10,
        batch_size: int = 5000,
        n_init: int = 10,
        random_state: int | None = None,
   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the stability of clustering results using a random baseline.

    This function uses K-Means clustering with varying numbers of clusters to evaluate the stability
    of the clustering results by comparing them against a random baseline. It can be used to
    determine the best number of clusters for the given data.

    Parameters
    ----------
    features : np.ndarray
        The input features for clustering.
    labels_true : np.ndarray
        The true labels for the data.
    n_clusters_list : list[int]
        A list of the number of clusters to evaluate.
    evaluation_fn : Callable
        The evaluation function to use for assessing clustering quality.
        Default is maximum_cluster_assignment_score.
    n_repetitions : int
        The number of times to repeat the evaluation for each cluster size. Default is 10.
    batch_size : int
        The batch size to use for K-Means clustering. Default is 5000.
    n_init : int
        The number of initializations to use for K-Means clustering. Default is 10.
    random_state : int | None
        The random state to use for reproducibility. Default is None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing the p-values, clustering scores, and random baseline scores.
    
    Examples
    --------
    >>> features = np.random.rand(100, 10)
    >>> labels_true = np.random.randint(0, 5, size=100)
    >>> n_clusters_list = [2, 3, 4, 5]
    >>> evaluate_clustering_stability(features, labels_true, n_clusters_list)
    array([0.05, 0.1 , 0.15, 0.2 ]), array([[0.8, 0.9, 0.7, 0.6],
          [0.7, 0.8, 0.6, 0.5],
          [0.6, 0.7, 0.5, 0.4],
          [0.5, 0.6, 0.4, 0.3]]), array([[0.1, 0.2, 0.3, 0.4],
          [0.2, 0.3, 0.4, 0.5],
          [0.3, 0.4, 0.5, 0.6],
          [0.4, 0.5, 0.6, 0.7]])
    """
    n = len(features)
    clustering_scores_list = []
    random_baseline_scores_list = []

    for n_clusters in n_clusters_list:

        clustering_scores = []
        random_baseline_scores = []

        for i in range(n_repetitions):
            sampled_indices = np.random.randint(0, n, size=n - int(np.sqrt(n)))
            features_sampled = features[sampled_indices]
            labels_sampled = labels_true[sampled_indices]

            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                init="random",
                n_init=n_init,
                batch_size=batch_size,
                random_state=random_state
            )
            kmeans.fit(features_sampled)
            clustering_scores.append(evaluation_fn(kmeans.labels_, labels_sampled))

            centroid_indices = np.random.randint(0, len(features_sampled), size=n_clusters)
            centroids = features_sampled[centroid_indices]
            random_labels = pairwise_distances_argmin(features_sampled, centroids)
            random_baseline_scores.append(evaluation_fn(random_labels, labels_sampled))

        clustering_scores_list.append(clustering_scores)
        random_baseline_scores_list.append(random_baseline_scores)

    clustering_scores_list = np.array(clustering_scores_list)
    random_baseline_scores_list = np.array(random_baseline_scores_list)

    statistics, pvalues = mannwhitneyu(clustering_scores_list, random_baseline_scores_list, alternative='greater', axis=1)

    return pvalues, clustering_scores_list, random_baseline_scores_list
