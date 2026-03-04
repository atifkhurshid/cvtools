"""
PyTorch wrapper for classification datasets.
"""

# Author: Atif Khurshid
# Created: 2026-03-03
# Modified: None
# Version: 1.0
# Changelog:
#     - 2026-03-03: Initial creation of PyTorch wrapper for classification datasets.

from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform

from ._base import _ClassificationBase


class PyTorchDataset(Dataset):
    def __init__(
            self,
            dataset: _ClassificationBase,
            transform: Optional[Transform] = None,
            target_transform: Optional[Callable] = None,
        ):
        """
        PyTorch wrapper class for classification datasets.

        Parameters
        ----------
        dataset: _ClassificationBase
            An instance of a classification dataset containing the data.
            It must implement the __getitem__ method to return (image, label) pairs.
        transform: torchvision.transforms.v2.Transform | None = None, optional
            Transform to apply to the images.
        target_transform: callable | None = None, optional
            Transform to apply to the labels.
        """
        super().__init__()
        
        self._dataset = dataset
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            The total number of samples in the dataset.
        """
        return len(self._dataset)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get an image and corresponding label from the dataset.
        The image is a tensor of shape (C, H, W) and the label is a float tensor of shape (1,).

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the image tensor and the label tensor.
        """
        image, label = self._dataset[index]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying dataset for
        any attributes not found in this wrapper class.
        """
        return getattr(self._dataset, name)
    