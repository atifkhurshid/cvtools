"""
Dataloader for saved feature maps.
"""

# Author: Atif Khurshid
# Created: 2025-08-08
# Modified: 2025-09-05
# Version: 2.0
# Changelog:
#     - 2025-09-05: Simplified class functionality to focus on loading saved features only.

from pathlib import Path

import numpy as np

from .._base import _ClassificationBase


class SavedFeaturesDataset():

    def __init__(
            self,
            dataset: _ClassificationBase,
            features_dir: str,
        ):
        """
        Dataset for loading saved feature maps and their corresponding labels.

        Parameters
        ----------
        dataset : _ClassificationBase
            Original dataset used to extract features.
        features_dir : str
            Directory containing the saved feature maps.
        
        Examples
        --------
        >>> import os
        >>> import torch
        >>> import numpy as np
        >>> from torchvision import transforms
        >>> from torch.utils.data import DataLoader
        >>> from cvtools.models.pytorch import extract_feature_maps
        >>> from cvtools.datasets.classification import SavedFeaturesDataset
        >>> from cvtools.datasets.classification.pytorch import ClassificationDatasetPT
        >>> # Define transformations
        >>> transform = transforms.Compose([
        ...     transforms.Resize((224, 224)),
        ...     transforms.ToTensor(),
        ... ])
        >>> # Load original dataset
        >>> dataset = ClassificationDatasetPT(
        ...     images_dir='path/to/images',
        ...     transform=transform,
        ... )
        >>> # Define model for feature extraction
        >>> # Extract and save feature maps
        >>> extract_feature_maps(
        ...     model=model,
        ...     dataset=dataset,
        ...     save_dir='path/to/save/features',
        ...     batch_size=32,
        ... )
        >>> # Load saved features dataset
        >>> features_dataset = SavedFeaturesDataset(
        ...     dataset=dataset,
        ...     features_dir='path/to/save/features',
        ... )
        >>> # Access a batch of features and labels
        >>> features, labels = features_dataset[0]
        >>> print(features.shape, labels.shape)
        (32, 512) (32,)
        """
        self.dataset = dataset
        self.features_dir = Path(features_dir)


    def __len__(self) -> int:
        """
        Returns the number of feature files in the dataset.
        """
        return len(self.dataset)


    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        """
        Returns a feature and corresponding label for a given index.

        Parameters
        ----------
        index : int
            Index of the feature and label files to retrieve.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the batch of feature tensors and the label tensors.
        """
        feature = np.load(self.features_dir / f"features_{index}.npy")
        label = self.dataset.labels[index]

        return feature, label


    def __getattr__(self, name: str):
        """
        Delegate attribute access to the underlying dataset.

        Parameters
        ----------
        name : str
            Name of the attribute to access.
        """
        return getattr(self.dataset, name)
