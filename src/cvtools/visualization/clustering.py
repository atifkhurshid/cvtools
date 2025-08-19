"""
Visualizations for clustering.
"""

# Author: Atif Khurshid
# Created: 2025-08-19
# Modified: None
# Version: 1.0
# Changelog:
#     - None

import numpy as np
import matplotlib.pyplot as plt

from ..evaluation.metrics import intra_cluster_variability
from ..evaluation.metrics import inter_cluster_variability


def visualize_cluster_variability(
        features: np.ndarray,
        labels: np.ndarray,
        metric: str = 'euclidean',
        figsize: tuple[int, int] = (16, 7),
    ):
    """
    Visualize the intra-cluster and inter-cluster variability.

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
    None
    """
    unique_labels = np.unique(labels)

    intra_vars = intra_cluster_variability(features, labels, metric=metric)
    inter_vars = inter_cluster_variability(features, labels, metric=metric)
    separability = inter_vars / (intra_vars + 1e-8)  # avoid division by zero


    fig, axes = plt.subplots(1, 2, figsize=figsize)

    scatter = axes[0].scatter(
        intra_vars, inter_vars, c=separability, cmap='viridis', s=100, edgecolor='k')
    
    for i, lbl in enumerate(unique_labels):
        axes[0].annotate(str(lbl), (intra_vars[i], inter_vars[i]), fontsize=10, ha='left', va='bottom')
    axes[0].set_xlabel('Within-Cluster Variability')
    axes[0].set_ylabel('Between-Cluster Variability')
    cbar = fig.colorbar(scatter, ax=axes[0])

    axes[1].bar(unique_labels, separability)
    axes[1].set_xlabel('Cluster ID')
    axes[1].set_ylabel('Separability Ratio')

    plt.tight_layout()
    plt.show()
