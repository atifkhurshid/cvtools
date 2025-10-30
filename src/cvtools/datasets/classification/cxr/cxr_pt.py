"""
PyTorch wrapper for NIH Chest X-Ray dataloader.
"""

# Author: Atif Khurshid
# Created: 2025-05-22
# Modified: 2025-10-30
# Version: 1.1
# Changelog:
#     - 2025-10-30: Updated arguments to match base class

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform
import torchvision.transforms.v2.functional as F

from .cxr import CXRDataset


class CXRDatasetPT(CXRDataset, Dataset):
    def __init__(
        self,
        root_dir: str,
        image_size: tuple[int, int] | None = None,
        preserve_aspect_ratio: bool = False,
        train: bool = True,
        binary: bool = True,
        transform: Transform | None = None,
        target_transform: callable | None = None,
    ):
        """
        PyTorch wrapper class for NIH Chest X-Ray dataset.

        Parameters
        ----------
        root_dir : str
            Path to the root directory of the dataset.
        image_size : tuple, optional
            Size of the images to be resized to (height, width). Default is None.
        train : bool, optional
            If True, load training/validation data. If False, load test data. Default is True.
        binary : bool, optional
            If True, convert labels to binary (0 for 'No Finding', 1 for 'Finding'). Default is True.
        transform: torchvision.transforms.v2.Transform | None = None, optional
            Transform to apply to the images.
        target_transform: callable | None = None, optional
            Transform to apply to the labels.

        Examples
        --------
        >>> import torch
        >>> from torchvision.transforms.v2 import ToDType
        >>> from torch.utils.data import DataLoader
        >>> from cvtools.datasets.classification.pytorch import CXRDatasetPT
        >>> dataset = CXRDatasetPT(root_dir='path/to/dataset', transform=ToDType(torch.float32))
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for X, y in dataloader:
        ...     # Process each batch of images and labels
        ...     pass

        """
        super().__init__(root_dir, image_size, preserve_aspect_ratio, train, binary)

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

