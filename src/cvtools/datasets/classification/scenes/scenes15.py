"""
Dataloader for 15-category Scenes dataset from Lazebnik et al. (2006)
Link: https://ieeexplore.ieee.org/document/1641019
"""

# Author: Atif Khurshid
# Created: 2022-09-05
# Modified: None
# Version: 1.0
# Changelog:
#     - None

from pathlib import Path

import numpy as np
from PIL import Image

from ....image import imresize
from .._base import _ClassificationBase


class Scenes15Dataset(_ClassificationBase):

    def __init__(
            self,
            root_dir: str,
            image_size: tuple[int, int] | None = None,
            preserve_aspect_ratio: bool = True,
        ):
        """
        15-category Scenes dataset loader.

        This class loads images and labels from the 15-category Scenes dataset.
        The dataset is expected to be organized with all images belonging to each class
        in separate subdirectories within the root directory.

        Parameters
        ---------- 
        root_dir : str
            Path to the root directory containing class subdirectories.
        image_size : tuple[int, int] | None, optional
            Size to which images will be resized. If None, images will not be resized. Default is None.
        preserve_aspect_ratio : bool, optional
            If True, images will be resized while preserving their aspect ratio. Default is True.

        Attributes
        ----------
        classes : list[str]
            List of class names (subdirectory names).
        labels : list[int]
            List of labels corresponding to the images.
        filenames : list[str]
            List of filenames of the images.

        Examples
        --------
        >>> from cvtools.datasets import Scenes15Dataset
        >>> dataset = Scenes15Dataset(root_dir='path/to/scenes15', image_size=(224, 224))
        >>> for img, label in dataset:
        ...     # Process each image and label
        ...     pass
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.preserve_aspect_ratio = preserve_aspect_ratio

        self.classes = [x.name for x in self.root_dir.iterdir() if not x.is_file()]

        self.labels = []
        self.filenames = []
        for c in self.classes:
            directory = self.root_dir / c
            filenames = [x.name for x in directory.iterdir() if x.suffix == '.jpg']
            labels = [c] * len(filenames)
            self.filenames.extend(filenames)
            self.labels.extend(labels)

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
        label = self.labels[index]
        path = self.root_dir / label / self.filenames[index]

        with Image.open(path) as image:
            if self.image_size:
                image = imresize(image, self.image_size, self.preserve_aspect_ratio)
            image = np.asarray(image, copy=True)

        return image, self.class_name_to_index(label)
