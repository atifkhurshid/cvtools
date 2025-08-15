"""
PyTorch Wrapper for saved features dataloader
"""

# Author: Atif Khurshid
# Created: 2025-08-08
# Modified: 2025-08-15
# Version: 1.1
# Changelog:
#     - 2025-08-15: Added support for transforms.

import torch
from torch.utils.data import Dataset

from .saved_features_dataset import SavedFeaturesDataset


class SavedFeaturesDatasetPT(SavedFeaturesDataset, Dataset):

    def __init__(self, *args, **kwargs):
        """
        PyTorch wrapper class for saved features dataset.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the SavedFeaturesDataset constructor.
        **kwargs : dict
            Keyword arguments passed to the SavedFeaturesDataset constructor.
            - transform: callable, optional
                Transform to apply to the features.
            - target_transform: callable, optional
                Transform to apply to the labels.
            
        Examples
        --------
        >>> dataset = SavedFeaturesDatasetPT("path/to/features")
        >>> len(dataset)
        10
        >>> features, labels = dataset[0]
        >>> features.shape
        (32, 512, 64, 64)
        >>> labels.shape
        (32,)
        """
        super().__init__(*args, **kwargs)

        self.transform = kwargs.pop("transform", None)
        self.target_transform = kwargs.pop("target_transform", None)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a batch of features and labels for a given index.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the batch of feature tensors and the label tensors.
        """
        features, labels = super().__getitem__(index)

        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        if self.transform is not None:
            features = self.transform(features)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return features, labels
