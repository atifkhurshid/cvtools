"""
PyTorch wrapper for 8-category Scenes dataloader.
"""

# Author: Atif Khurshid
# Created: 2025-09-03
# Modified: 2025-10-30
# Version: 1.1
# Changelog:
#     - 2025-10-30: Updated arguments to match base class

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform
import torchvision.transforms.v2.functional as F

from .scenes8 import Scenes8Dataset


class Scenes8DatasetPT(Scenes8Dataset, Dataset):
    def __init__(
            self,
            root_dir: str,
            image_size: tuple[int, int] | None = None,
            preserve_aspect_ratio: bool = True,
            transform: Transform | None = None,
            target_transform: callable | None = None,
        ):
        """
        PyTorch wrapper class for the 8-category Scenes dataset.

        Parameters
        ----------
        root_dir : str
            Path to the root directory of the dataset.
        image_size : tuple, optional
            Size of the images to be resized to (height, width). Default is None.
        preserve_aspect_ratio : bool, optional
            If True, preserve the aspect ratio of the images when resizing. Default is True.
        transform: torchvision.transforms.v2.Transform | None = None, optional
            Transform to apply to the images.
        target_transform: callable | None = None, optional
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
