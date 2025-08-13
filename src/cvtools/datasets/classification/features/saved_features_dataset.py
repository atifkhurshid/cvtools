"""
Dataloader for saved feature maps.
"""

# Author: Atif Khurshid
# Created: 2025-08-08
# Modified: None
# Version: 1.0
# Changelog:
#     - None

from pathlib import Path

import numpy as np

from .._base import _ClassificationBase


class SavedFeaturesDataset(_ClassificationBase):

    def __init__(
            self,
            features_dir: str,
            class_names: list[str] | None = None
        ):
        """
        Dataset for loading saved feature maps and their corresponding labels.

        Parameters
        ----------
        features_dir : str
            Directory containing the saved feature maps and labels.
        class_names : list[str] | None, optional
            List of class names corresponding to the labels. If None, classes will be inferred from the labels.
        
        Attributes
        ----------
        feature_files : list
            List of paths to the feature files.
        label_files : list
            List of paths to the label files.
        
        Examples
        --------
        >>> dataset = SavedFeatureMaps("path/to/features")
        >>> len(dataset)
        10
        >>> features, labels = dataset[0]
        >>> features.shape
        (32, 512, 64, 64)
        >>> labels.shape
        (32,)
        """
        self.features_dir = Path(features_dir)
        self.feature_files = sorted(self.features_dir.glob("features_batch_*.npy"))
        self.label_files = sorted(self.features_dir.glob("labels_batch_*.npy"))

        assert len(self.feature_files) == len(self.label_files), \
            "Number of feature files and label files must match."

        self.labels = []
        for label_file in self.label_files:
            labels = np.load(label_file)
            self.labels.extend(labels.tolist())
        self.labels = np.array(self.labels)

        unique_labels = np.unique(self.labels)
        if class_names is not None:
            assert len(class_names) == len(unique_labels), \
                "Number of class names must match number of unique labels."
            self.classes = class_names
        else:
            self.classes = [f"Class {i}" for i in unique_labels]

        self.__initialize__()


    def __len__(self) -> int:
        """
        Returns the number of feature files in the dataset.
        """
        return len(self.feature_files)


    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a batch of feature and label tensors for a given index.

        Parameters
        ----------
        index : int
            Index of the feature and label files to retrieve.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the batch of feature tensors and the label tensors.
        """
        features = np.load(self.feature_files[index])
        labels = np.load(self.label_files[index])

        return features, labels
