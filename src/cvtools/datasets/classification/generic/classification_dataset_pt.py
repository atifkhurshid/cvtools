"""
PyTorch Wrapper for generic image classification dataloader
"""

# Author: Atif Khurshid
# Created: 2022-12-18
# Modified: 2025-10-30
# Version: 1.1
# Changelog:
#     - 2025-06-18: Updated documentation and type hints.
#     - 2025-10-30: Updated arguments to match base class.

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform
import torchvision.transforms.v2.functional as F

from .classification_dataset import ClassificationDataset


class ClassificationDatasetPT(ClassificationDataset, Dataset):

    def __init__(
            self,
            root_dir: str,
            exts: list[str] = ['.jpg', '.png'],
            image_mode: str | int = 'RGB',
            image_size: tuple[int, int] | None = None,
            preserve_aspect_ratio: bool = True,
            transform: Transform | None = None,
            target_transform: callable | None = None,
        ):
        """
        PyTorch wrapper class for generic image classification dataset.

        Parameters
        ----------
        root_dir : str
            Path to the root directory containing class subdirectories.
        exts : list[str], optional
            List of file extensions to consider as valid images. Default is ['.jpg', '.png'].
        image_mode : str | int, optional
            Mode to read images. Can be 'RGB', 'GRAY', or a cv2.IMREAD_... flag. Default is 'RGB'.
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
        super().__init__(root_dir, exts, image_mode, image_size, preserve_aspect_ratio)

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
        id = self.ids[index]

        image, label = super().__getitem__(id)
        image = F.to_image(image)
        label = torch.tensor(label)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
