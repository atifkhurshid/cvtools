"""
PyTorch wrapper for 8-category Scenes dataloader.
"""

# Author: Atif Khurshid
# Created: 2025-09-03
# Modified: None
# Version: 1.0
# Changelog:
#     - None

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ToImage

from .scenes8 import Scenes8Dataset


class Scenes8DatasetPT(Scenes8Dataset, Dataset):
    def __init__(self, *args, **kwargs):
        """
        PyTorch wrapper class for the 8-category Scenes dataset.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the Scenes8Dataset constructor.
        **kwargs : dict
            Keyword arguments passed to the Scenes8Dataset constructor.
            - transform: torchvision.transforms.v2.Transform, optional
                Transform to apply to the images.
            - target_transform: callable, optional
                Transform to apply to the labels.

        Examples
        --------
        >>> from cvtools.datasets import Scenes8DatasetPT
        >>> dataset = Scenes8DatasetPT(root_dir='path/to/scenes', transform=ToDType(torch.float32))
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
