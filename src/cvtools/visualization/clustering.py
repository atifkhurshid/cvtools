"""
Visualizations for clustering.
"""

# Author: Atif Khurshid
# Created: 2025-08-19
# Modified: 2025-10-16
# Version: 1.5
# Changelog:
#     - 2025-08-20: Added clustering stability visualization.
#     - 2025-08-21: Updated cluster variability visualization.
#     - 2025-08-27: Updated cluster variability visualization.
#     - 2025-09-02: Changed separability index calculations.
#     - 2025-10-16: Added options to save visualizations to files.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from ..evaluation.metrics import intra_cluster_variability
from ..evaluation.metrics import inter_cluster_variability


def visualize_cluster_variability(
        features: np.ndarray,
        labels: np.ndarray,
        metric: str = 'euclidean',
        figsize: tuple[int, int] = (14, 5),
        cmap: str = 'RdYlGn',
        bar_width: float = 0.3,
        variability_limits: tuple[float, float] | None = None,
        save_path: str | None = None,
        save_dpi: int = 600,
        save_format: str = 'png',
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
    figsize : tuple[int, int], default=(14, 5)
        Figure size for the plot.
    cmap : str, default='viridis'
        Colormap for the scatter plot.
    bar_width : float, default=0.3
        Width of the bars in the bar plot.
    variability_limits : tuple[float, float] | None, optional
        Limits for the variability plot. Default is None.
    save_path : str | None, optional
        Path to save the figure, if None the figure is not saved, default is None.
    save_dpi : int, optional
        Dots per inch for saving the figure, default is 600.
    save_format : str, optional
        Format to save the figure, default is 'png'.
        
    Returns
    -------
    None
    """
    unique_labels = np.unique(labels)

    intra_vars = intra_cluster_variability(features, labels, metric=metric)
    inter_vars = inter_cluster_variability(features, labels, metric=metric)

    separability = (inter_vars - intra_vars) / (inter_vars + intra_vars + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if variability_limits is None:
        variability_limits = 0, max(np.max(intra_vars), np.max(inter_vars)) * 1.1

    norm = plt.Normalize(-1, 1)

    scatter = axes[0].scatter(
        intra_vars, inter_vars, c=separability, cmap=cmap, norm=norm, s=500, edgecolor='k')

    for i, lbl in enumerate(unique_labels):
        axes[0].annotate(str(lbl), (intra_vars[i], inter_vars[i]), color="k", fontsize=10, ha='center', va='center')
    axes[0].set_xlabel('Within-Cluster Variability')
    axes[0].set_ylabel('Between-Cluster Variability')
    axes[0].set_xlim(variability_limits)
    axes[0].set_ylim(variability_limits)


    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0])
    colors = plt.get_cmap(cmap)(norm(separability))

    axes[1].bar(unique_labels.astype(str), np.maximum(0, separability), width=bar_width, color=colors, edgecolor='k')
    axes[1].set_xlabel('Cluster ID')
    axes[1].set_ylabel('Separability Index')
    axes[1].set_ylim(0, 1.1)

    if save_path is not None:
        plt.savefig(save_path, dpi=save_dpi, format=save_format)

    plt.show()
    

def visualize_clustering_stability(
        clustering_scores: np.ndarray,
        random_baseline_scores: np.ndarray,
        pvalues: np.ndarray,
        n_clusters_list: list | np.ndarray | None = None,
        box_width: float = 0.15,
        figsize: tuple[int, int] = (10, 6),
        save_path: str | None = None,
        save_dpi: int = 600,
        save_format: str = 'png',
    ):
    """import matplotlib.pyplot as plt
    Visualize clustering stability using boxplots.

    This function creates boxplots to compare clustering scores against a random baseline,
    where the scores have been obtained by multiple runs of the clustering algorithm.

    Parameters
    ----------
    clustering_scores : np.ndarray
        Clustering scores for each number of clusters.
    random_baseline_scores : np.ndarray
        Random baseline scores for each number of clusters.
    pvalues : np.ndarray
        p-values for statistical significance.
    n_clusters_list : list | np.ndarray | None, optional
        List of number of clusters to consider. Default is None
    box_width : float, optional
        Width of the boxes in the boxplot. Default is 0.15
    figsize : tuple[int, int], optional
        Size of the figure. Default is (10, 6).
    save_path : str | None, optional
        Path to save the figure, if None the figure is not saved, default is None.
    save_dpi : int, optional
        Dots per inch for saving the figure, default is 600.
    save_format : str, optional
        Format to save the figure, default is 'png'.
    
    Examples
    --------
    >>> clustering_scores = np.random.rand(10, 5)
    >>> random_baseline_scores = np.random.rand(10, 5)
    >>> pvalues = np.random.rand(5)
    >>> visualize_clustering_stability(clustering_scores, random_baseline_scores, pvalues)
    ...
    """

    if n_clusters_list is None:
        n_clusters_list = np.arange(1, len(clustering_scores) + 1)
    n_clusters_list = np.array(n_clusters_list)

    fig, ax = plt.subplots(1, figsize=figsize)

    positions_clustering = n_clusters_list - box_width / 2
    positions_random = n_clusters_list + box_width / 2

    bp1 = ax.boxplot(
        clustering_scores.T,
        positions=positions_clustering,
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor='indianred', color='maroon'),
        medianprops=dict(color='maroon'),
        whiskerprops=dict(color='maroon'),
        capprops=dict(color='maroon'),
        flierprops=dict(markerfacecolor='indianred', marker='o', color='maroon')
    )

    bp2 = ax.boxplot(
        random_baseline_scores.T,
        positions=positions_random,
        widths=box_width,
        patch_artist=True,
        boxprops=dict(facecolor='steelblue', color='midnightblue'),
        medianprops=dict(color='midnightblue'),
        whiskerprops=dict(color='midnightblue'),
        capprops=dict(color='midnightblue'),
        flierprops=dict(markerfacecolor='steelblue', marker='o', color='steelblue')
    )

    for n_clusters, p in zip(n_clusters_list, pvalues):
        ax.text(n_clusters, 1.05, f"p={p:.3f}", ha='center', va='bottom', fontsize=12, color='black')

    ax.set_ylim(0, 1.0)    
    ax.set_xticks(n_clusters_list, n_clusters_list)
    ax.set_xlabel('Number of clusters (K)')
    ax.set_ylabel('Score')
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['KMeans', 'Random Baseline'], loc='upper right')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=save_dpi, format=save_format)

    plt.show()
