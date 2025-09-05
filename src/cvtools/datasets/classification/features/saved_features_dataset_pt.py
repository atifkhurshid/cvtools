"""
PyTorch Wrapper for saved features dataloader
"""

# Author: Atif Khurshid
# Created: 2025-08-08
# Modified: 2025-09-05
# Version: 1.1
# Changelog:
#     - 2025-08-15: Added support for transforms.
#     - 2025-09-05: Updated according to changes in SavedFeaturesDataset.

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
        >>> dataset = SavedFeaturesDatasetPT(images_dataset, "path/to/features")
        >>> len(dataset)
        10
        >>> feature, label = dataset[0]
        >>> feature.shape
        (512, 64, 64)
        >>> label
        3
        """
        self.transform = kwargs.pop("transform", None)
        self.target_transform = kwargs.pop("target_transform", None)

        super().__init__(*args, **kwargs)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get the feature map and label for a given index.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the feature tensor and the label tensor.
        """
        feature, label = super().__getitem__(index)

        feature = torch.tensor(feature, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform is not None:
            feature = self.transform(feature)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return feature, label
