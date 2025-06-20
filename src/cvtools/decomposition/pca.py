"""
Package: decomposition.pca
Requirements:
    - sklearn
Use: 
    - from decomposition.pca import *
Methods:
    - None
Class:
    - PCA

Author: Atif Khurshid

Created: 2025-06-02
Modified: None
Version: 1.0

Changelog:
    - None
"""
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA


class PCA():

    def __init__(
            self,
            n_components: int | str = "best",
            explained_variance_threshold: float = 0.95,
        ):

        self.pca = None
        self.explained_variance_threshold = explained_variance_threshold
        self.selected_components = None
    

    def __call__(self, x):
        self.transform(x)
    

    def fit(self, x, batch_size=1000):

        n_samples, n_features = x.shape
        self.pca = IncrementalPCA(n_components=n_features)
        self.selected_components = None

        batch_size = max(n_features + 1, batch_size)  # Ensure batch size is at least n_features + 1
        for i in tqdm(range(0, n_samples, batch_size), total=n_samples // batch_size):
            batch = x[i:i + batch_size]
            self.pca.partial_fit(batch)

        cumulative_variance = self.pca.explained_variance_ratio_.cumsum()
        self.selected_components = self.pca.components_[cumulative_variance <= self.explained_variance_threshold]


    def transform(self, x):
        if self.pca:
            x -= self.pca.mean_
            x = x @ self.selected_components.T
            return x
        else:
            raise Exception("Error: PCA model is not fitted yet.")
    