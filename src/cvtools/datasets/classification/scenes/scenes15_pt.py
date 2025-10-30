"""
PyTorch wrapper for 15-category Scenes dataloader.
"""

# Author: Atif Khurshid
# Created: 2025-09-05
# Modified: 2025-10-30
# Version: 1.1
# Changelog:
#     - 2025-10-30: Updated arguments to match base class

from typing import Callable, Optional

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform
import torchvision.transforms.v2.functional as F

from .scenes15 import Scenes15Dataset


class Scenes15DatasetPT(Scenes15Dataset, Dataset):
    def __init__(
            self,
            root_dir: str,
            image_size: Optional[tuple[int, int]] = None,
            preserve_aspect_ratio: bool = True,
            transform: Optional[Transform] = None,
            target_transform: Optional[Callable] = None,
        ):
        """
        PyTorch wrapper class for the 15-category Scenes dataset.

        Parameters
        ----------
        root_dir : str
            Path to the root directory containing class subdirectories.
        image_size : tuple[int, int] | None, optional
            Size to which images will be resized. If None, images will not be resized. Default is None.
        preserve_aspect_ratio : bool, optional
            If True, images will be resized while preserving their aspect ratio. Default is True.
        transform: torchvision.transforms.v2.Transform | None = None, optional
            Transform to apply to the images.
        target_transform: callable | None = None, optional
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
        super().__init__(root_dir, image_size, preserve_aspect_ratio)

        self.transform = transform
        self.target_transform = target_transform


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

        image = F.to_image(image)
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
