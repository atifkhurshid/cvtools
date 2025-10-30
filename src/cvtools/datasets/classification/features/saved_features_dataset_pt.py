"""
PyTorch Wrapper for saved features dataloader
"""

# Author: Atif Khurshid
# Created: 2025-08-08
# Modified: 2025-10-30
# Version: 1.2
# Changelog:
#     - 2025-08-15: Added support for transforms.
#     - 2025-09-05: Updated according to changes in SavedFeaturesDataset.
#     - 2025-10-30: Updated arguments to match base class

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform

from .saved_features_dataset import SavedFeaturesDataset


class SavedFeaturesDatasetPT(SavedFeaturesDataset, Dataset):

    def __init__(
            self,
            dataset: Dataset,
            features_dir: str,
            transform: Transform | None = None,
            target_transform: callable | None = None,
        ):
        """
        PyTorch wrapper class for saved features dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Original dataset used to extract features.
        features_dir : str
            Directory containing the saved feature maps.
        transform: torchvision.transforms.v2.Transform | None = None, optional
            Transform to apply to the features.
        target_transform: callable | None = None, optional
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
        super().__init__(dataset, features_dir)

        self.transform = transform
        self.target_transform = target_transform


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
