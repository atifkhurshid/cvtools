"""
Generic dataloader for image classification tasks.
"""

# Author: Atif Khurshid
# Created: 2022-12-18
# Modified: 2025-10-30
# Version: 1.1
# Changelog:
#     - 2025-06-18: Updated documentation and type hints.
#     - 2025-10-30: Refactored image loading to use imread function.

from pathlib import Path
from typing import Optional, Union

import numpy as np

from ....image import imread
from .._base import _ClassificationBase


class ClassificationDataset(_ClassificationBase):

    def __init__(
            self,
            root_dir: str,
            exts: list[str] = ['.jpg', '.png'],
            image_mode: Union[str, int] = 'RGB',
            image_size: Optional[tuple[int, int]] = None,
            preserve_aspect_ratio: bool = True,
        ):
        """
        Generic classification dataset class.

        This class loads images from a directory structure where each subdirectory
        corresponds to a class label.

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

        Attributes
        ----------
        classes : list[str]
            List of class names (subdirectory names).
        class2label : dict[str, int]
            Mapping from class names to integer labels.
        label2class : dict[int, str]
            Mapping from integer labels to class names.
        labels : list[int]
            List of labels corresponding to the images.
        filenames : list[str]
            List of filenames of the images.
        ids : np.ndarray
            Array of indices for the images, used for indexing into the dataset.

        Examples
        -----
        >>> dataset = ClassificationDataset(root_dir='path/to/dataset', image_size=(224, 224))
        >>> print(len(dataset))  # Number of images in the dataset
        >>> image, label = dataset[0]  # Get the first image and its label
        >>> print(image.shape, label)
        >>> for image, label in dataset:
        ...     # Process each image and label
        ...     pass
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.image_mode = image_mode

        self.classes = [x.name for x in self.root_dir.iterdir() if not x.is_file()]

        self.labels = []
        self.filenames = []
        for i, c in enumerate(self.classes):
            directory = self.root_dir / c
            filenames = [x.name for x in directory.iterdir() if x.suffix in exts]
            labels = [c] * len(filenames)
            self.filenames.extend(filenames)
            self.labels.extend(labels)

        self.ids = np.arange(len(self.filenames))

        self.__initialize__()


    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        """
        Get an image and corresponding label from the dataset.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        tuple[np.ndarray, int]
            A tuple containing the image as a NumPy array and its corresponding label.
        """
        id = self.ids[index]
        label = self.labels[id]
        image_path = str(self.root_dir / label / self.filenames[id])

        image = imread(
            image_path,
            mode=self.image_mode,
            size=self.image_size,
            preserve_aspect_ratio=self.preserve_aspect_ratio,
        )

        label = self.class_name_to_index(label)

        return image, label
