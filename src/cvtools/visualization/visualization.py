"""
PCA visualization module.
"""

# Author: Atif Khurshid
# Created: 2025-06-16
# Modified: 2025-08-04
# Version: 1.1
# Changelog:
#     - 2025-08-04: Add support for t-SNE visualization.

import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import IncrementalPCA


plt.style.use('seaborn-v0_8-notebook')

def visualize_features(
        mode: str,
        features: list[np.ndarray] | np.ndarray,
        labels: list | np.ndarray,
        class_names: list[str],
        n_components: int = 2,
        batch_size: int = 1000,
        perplexity: float = 30.0,
        figsize: tuple[int, int] = (10, 8),
    ):
    """
    Visualize high-dimensional features using PCA, or t-SNE.

    Parameters
    ----------
    mode : str
        Visualization mode, either 'pca', or 'tsne'.
    features : list of np.ndarray or np.ndarray
        High-dimensional features to be reduced.
    labels : list or np.ndarray
        Labels corresponding to the features, used for coloring the points.
    class_names : list of str
        Names of the classes corresponding to the labels.
    n_components : int, optional
        Number of principal components to keep, default is 2.
    batch_size : int, optional
        Batch size for IncrementalPCA, default is 1000.
    perplexity : float, optional
        Perplexity parameter for t-SNE, default is 30.
    figsize : tuple of int, optional
        Size of the figure for visualization, default is (10, 8).
    
    Examples
    --------
    >>> import numpy as np
    >>> from cvtools.visualization import visualize_features
    >>> features = np.random.rand(100, 50)  # 100 samples, 50 features
    >>> labels = np.random.randint(0, 5, size=100)  # 5 classes
    >>> class_names = [f'Class {i}' for i in range(5)]
    >>> visualize_features('pca', features, labels, class_names, n_components=2, batch_size=10, figsize=(12, 10))
    """
    assert n_components in [2, 3], "n_components must be either 2 or 3 for visualization."

    if mode == 'pca':
        pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        reduced_features = pca.fit_transform(features)

        print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_)}")

    elif mode == 'tsne':
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        reduced_features = tsne.fit_transform(features)

    else:
        raise ValueError("Invalid mode. Choose either 'pca' or 'tsne'.")

    fig = plt.figure(1, figsize=figsize)

    if n_components == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap="tab20")

    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d', elev=-150, azim=110)

        scatter = ax.scatter(
            reduced_features[:, 0],
            reduced_features[:, 1],
            reduced_features[:, 2],
            c=labels,
            cmap="tab20"
        )
        ax.set_zlabel('PC 3')
        ax.zaxis.set_ticklabels([])

    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax.legend(
        scatter.legend_elements(num=None)[0],
        class_names,
        loc="center left",
        title="Classes",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
    )

    plt.show()
