"""
Customized PCA module.
"""

# Author: Atif Khurshid
# Created: 2025-06-02
# Modified: 2025-08-01
# Version: 1.3
# Changelog:
#     - 2025-06-21: Change fit function to directly use fit function of IncrementalPCA
#     - 2025-06-21: Add partial fit functionality
#     - 2025-08-01: Add documentation and type hints.

from typing import Optional

import numpy as np
from sklearn.decomposition import IncrementalPCA


class PCA():

    def __init__(
            self,
            explained_variance_threshold: float = 0.95,
        ):
        """
        Principal Component Analysis with dynamic component selection.

        This implementation allows for incremental fitting and dynamic selection of
        principal components based on the explained variance ratio.

        Parameters
        ----------
        explained_variance_threshold : float, optional
            The cumulative explained variance ratio threshold for selecting components.
            Default is 0.95, meaning components are selected until 95% of the variance
            is explained.

        Attributes
        ----------
        pca : IncrementalPCA
            The IncrementalPCA instance used for fitting and transforming data.
        selected_components : np.ndarray
            The selected principal components after fitting the model. Initially None.
        
        Examples
        --------
        >>> pca = PCA(explained_variance_threshold=0.95)
        >>> data = np.random.rand(100, 20)  # 100 samples, 20 features
        >>> pca.fit(data, batch_size=50)
        >>> transformed_data = pca.transform(data)
        >>> print(transformed_data.shape)  # Should be (100, n_selected_components)

        >>> for i in range(0, 100, 50):
        ...     pca.partial_fit(data[i:i+50])
        >>> transformed_data = pca.transform(data)
        >>> print(transformed_data.shape)  # Should be (100, n_selected_components)

        >>> # To use PCA as a callable
        >>> transformed_data = pca(data)
        >>> print(transformed_data.shape)  # Should be (100, n_selected_components)
        """
        self.pca = None
        self.explained_variance_threshold = explained_variance_threshold
        self.selected_components = None
    

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Transform the input data using the fitted PCA model.

        Parameters
        ----------
        x : np.ndarray
            Input data to be transformed. Should be a 2D array where rows are samples
            and columns are features.

        Returns
        -------
        np.ndarray
            Transformed data after applying PCA. The shape will be (n_samples, n_selected_components).
        """
        return self.transform(x)
    

    def fit(
            self,
            x: np.ndarray,
            batch_size: Optional[int] = None,
        ):
        """
        Fit the PCA model to the input data.

        Parameters
        ----------
        x : np.ndarray
            Input data to fit the PCA model. Should be a 2D array where rows are samples
            and columns are features.
        batch_size : int | None, optional
            The batch size to use for incremental fitting. If None, the entire dataset
            will be used at once. Default is None.
        """
        n_samples, n_features = x.shape
        self.pca = IncrementalPCA(n_components=n_features, batch_size=batch_size)
        self.pca.fit(x)
        self._finalize()


    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Transform the input data using the fitted PCA model.

        Parameters
        ----------
        x : np.ndarray
            Input data to be transformed. Should be a 2D array where rows are samples
            and columns are features.

        Returns
        -------
        np.ndarray
            Transformed data after applying PCA. The shape will be (n_samples, n_selected_components).
        """
        if self.pca:
            x -= self.pca.mean_
            x = x @ self.selected_components.T
            return x
        else:
            raise Exception("Error: PCA model is not fitted yet.")


    def partial_fit(
            self,
            x: np.ndarray,
            batch_size: Optional[int] = None,
        ):
        """
        Incrementally fit the PCA model to the input data.

        Parameters
        ----------
        x : np.ndarray
            Input data to fit the PCA model. Should be a 2D array where rows are samples
            and columns are features.
        batch_size : int | None, optional
            The batch size to use for incremental fitting. If None, the entire dataset
            will be used at once. Default is None.
        """
        if not self.pca:
            n_samples, n_features = x.shape
            self.pca = IncrementalPCA(n_components=n_features, batch_size=batch_size)
        self.pca.partial_fit(x)
        self._finalize()
    

    def _finalize(self):
        """
        Finalize the PCA model by selecting components based on the explained variance threshold.
        """
        if self.pca:
            cumulative_variance = self.pca.explained_variance_ratio_.cumsum()
            self.selected_components = self.pca.components_[cumulative_variance <= self.explained_variance_threshold]
        else:
            raise Exception("Error: PCA model is not fitted yet.")
        