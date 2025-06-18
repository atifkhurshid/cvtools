"""
Random sampling utilities.
"""

# Author: Atif Khurshid
# Created: 2025-06-16
# Modified: None
# Version: 1.0
# Changelog:
#     - None

import numpy as np


def stratified_sampling_by_class(
        features: list[np.ndarray] | np.ndarray,
        labels: list | np.ndarray,
        n_samples: int = 10,
        seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Stratfied sampling of features and labels, based on class.
    N random samples are taken from each class.

    Parameters
    ----------
    features : list of np.ndarray or np.ndarray
        High-dimensional features to be sampled.
    labels : list or np.ndarray
        Labels corresponding to the features, used for grouping samples.
    n_samples : int, optional
        Number of samples to take from each class, default is 10.
    seed : int, optional
        Random seed for reproducibility, default is 42.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Sampled features and corresponding labels.
    
    Examples
    --------
    >>> import numpy as np
    >>> from cvtools.visualization import pca_visualization
    >>> from cvtools.utils import stratified_sampling_by_class
    >>> features = np.random.rand(100, 50)  # 100 samples, 50 features
    >>> labels = np.random.randint(0, 5, size=100)  # 5 classes
    >>> sampled_features, sampled_labels = stratified_sampling_by_class(features, labels, n_samples=10, seed=42)
    >>> classes = [f'Class {i}' for i in range(5)]
    >>> pca_visualization(sampled_features, sampled_labels, class_names=classes, n_components=2, batch_size=10, figsize=(12, 10))
    """
    rng = np.random.default_rng(seed=seed)

    features_sampled = []
    labels_sampled = []
    for label in np.unique(labels):
        idx = np.where(labels == label)[0]
        idx = rng.choice(idx, size=n_samples, replace=False)
        features_sampled.append(features[idx])
        labels_sampled.append(labels[idx])

    return np.concatenate(features_sampled), np.concatenate(labels_sampled)

