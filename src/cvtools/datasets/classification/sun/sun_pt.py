"""
PyTorch wrapper for Princeton SUN dataloader.
"""

# Author: Atif Khurshid
# Created: 2025-05-23
# Modified: None
# Version: 1.0
# Changelog:
#     - None

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ToImage

from .sun import SUNDataset


class SUNDatasetPT(SUNDataset, Dataset):
    def __init__(self, *args, **kwargs):
        """
        PyTorch wrapper class for the Princeton SUN dataset.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the SUNDataset constructor.
        **kwargs : dict
            Keyword arguments passed to the SUNDataset constructor.
            - transform: torchvision.transforms.v2.Transform, optional
                Transform to apply to the images.
            - target_transform: callable, optional
                Transform to apply to the labels.

        Examples
        --------
        >>> import torch
        >>> from torchvision.transforms.v2 import ToDType
        >>> from torch.utils.data import DataLoader
        >>> from cvtools.datasets.classification.pytorch import SUNDatasetPT
        >>> dataset = SUNDatasetPT(root_dir='path/to/sun', transform=ToDType(torch.float32))
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
