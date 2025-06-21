"""
Customized PCA module.
"""

# Author: Atif Khurshid
# Created: 2025-06-02
# Modified: 2025-06-21
# Version: 1.1
# Changelog:
#     - 2025-06-21: Change fit function to directly use fit function of IncrementalPCA

import numpy as np
from sklearn.decomposition import IncrementalPCA


class PCA():
    def __init__(
            self,
            explained_variance_threshold: float = 0.95,
        ):
        self.pca = None
        self.explained_variance_threshold = explained_variance_threshold
        self.selected_components = None
    

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.transform(x)
    

    def fit(
            self,
            x: np.ndarray,
            batch_size: int | None = None,
        ):
        n_samples, n_features = x.shape
        self.pca = IncrementalPCA(n_components=n_features, batch_size=batch_size)
        self.pca.fit(x)
        cumulative_variance = self.pca.explained_variance_ratio_.cumsum()
        self.selected_components = self.pca.components_[cumulative_variance <= self.explained_variance_threshold]


    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.pca:
            x -= self.pca.mean_
            x = x @ self.selected_components.T
            return x
        else:
            raise Exception("Error: PCA model is not fitted yet.")
    