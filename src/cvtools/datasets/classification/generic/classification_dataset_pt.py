"""
PyTorch Wrapper for generic image classification dataloader
"""

# Author: Atif Khurshid
# Created: 2022-12-18
# Modified: 2025-06-18
# Version: 1.0
# Changelog:
#     - 2025-06-18: Updated documentation and type hints.

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ToImage

from .classification_dataset import ClassificationDataset


class ClassificationDatasetPT(ClassificationDataset, Dataset):

    def __init__(self, *args, **kwargs):
        """
        PyTorch wrapper class for generic image classification dataset.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the ClassificationDataset constructor.
        **kwargs : dict
            Keyword arguments passed to the ClassificationDataset constructor.
            - transform: torchvision.transforms.v2.Transform, optional
                Transform to apply to the images.
            - target_transform: callable, optional
                Transform to apply to the labels.

        Examples
        --------
        >>> import torch
        >>> from torchvision.transforms.v2 import ToDType
        >>> from torch.utils.data import DataLoader
        >>> from cvtools.datasets.classification.pytorch import ClassificationDatasetPT
        >>> dataset = ClassificationDatasetPT(root_dir='path/to/dataset', transform=ToDType(torch.float32))
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for X, y in dataloader:
        ...     # Process each batch of images and labels
        ...     pass
        """
        super().__init__(*args, **kwargs)

        self.transform = kwargs.get("transform", None)
        self.target_transform = kwargs.get("target_transform", None)


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
        id = self.ids[index]

        image, label = super().__getitem__(id)
        image = ToImage()(image)
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
