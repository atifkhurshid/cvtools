"""
PCA visualization module.
"""

# Author: Atif Khurshid
# Created: 2025-06-16
# Modified: None
# Version: 1.0
# Changelog:
#     - None

import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA


plt.style.use('seaborn-v0_8-notebook')

def pca_visualization(
        features: list[np.ndarray] | np.ndarray,
        labels: list | np.ndarray,
        class_names: list[str],
        n_components: int = 2,
        batch_size: int = 1000,
        figsize: tuple[int, int] = (10, 8),
    ):
    """
    Visualize high-dimensional features using PCA.

    Parameters
    ----------
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
    figsize : tuple of int, optional
        Size of the figure for visualization, default is (10, 8).
    
    Examples
    --------
    >>> import numpy as np
    >>> from cvtools.visualization import pca_visualization
    >>> features = np.random.rand(100, 50)  # 100 samples, 50 features
    >>> labels = np.random.randint(0, 5, size=100)  # 5 classes
    >>> class_names = [f'Class {i}' for i in range(5)]
    >>> pca_visualization(features, labels, class_names, n_components=2, batch_size=10, figsize=(12, 10))
    """
    assert n_components in [2, 3], "n_components must be either 2 or 3 for visualization."

    pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    reduced_features = pca.fit_transform(features)

    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_)}")

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
        scatter.legend_elements(num=len(class_names))[0],
        class_names,
        loc="center left",
        title="Classes",
        bbox_to_anchor=(1, 0.5),
        ncol=1,
    )

    plt.show()
