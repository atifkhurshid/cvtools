"""
PyTorch Wrapper for saved features dataloader
"""

# Author: Atif Khurshid
# Created: 2022-08-08
# Modified: None
# Version: 1.0
# Changelog:
#     - None

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

        return features, labels
