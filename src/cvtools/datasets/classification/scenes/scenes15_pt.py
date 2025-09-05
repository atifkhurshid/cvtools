"""
PyTorch wrapper for 15-category Scenes dataloader.
"""

# Author: Atif Khurshid
# Created: 2025-09-05
# Modified: None
# Version: 1.0
# Changelog:
#     - None

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ToImage

from .scenes15 import Scenes15Dataset


class Scenes15DatasetPT(Scenes15Dataset, Dataset):
    def __init__(self, *args, **kwargs):
        """
        PyTorch wrapper class for the 15-category Scenes dataset.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the Scenes15Dataset constructor.
        **kwargs : dict
            Keyword arguments passed to the Scenes15Dataset constructor.
            - transform: torchvision.transforms.v2.Transform, optional
                Transform to apply to the images.
            - target_transform: callable, optional
                Transform to apply to the labels.

        Examples
        --------
        >>> from cvtools.datasets import Scenes15DatasetPT
        >>> dataset = Scenes15DatasetPT(root_dir='path/to/scenes', transform=ToDType(torch.float32))
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for X, y in dataloader:
        ...     # Process each batch of images and labels
        ...     pass
        """
        self.transform = kwargs.pop("transform", None)
        self.target_transform = kwargs.pop("target_transform", None)

        super().__init__(*args, **kwargs)


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
        image, label = super().__getitem__(index)

        image = ToImage()(image)
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
