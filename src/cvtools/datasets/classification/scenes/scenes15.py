"""
Dataloader for 15-category Scenes dataset from Lazebnik et al. (2006)
Link: https://ieeexplore.ieee.org/document/1641019
"""

# Author: Atif Khurshid
# Created: 2022-09-05
# Modified: 2026-03-03
# Version: 1.2
# Changelog:
#     - 2026:02-10: Used custom imread function.
#     - 2026-03-03: Refactored code to use new image processing functions.

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ....image import imread
from .._base import _ClassificationBase


class Scenes15Dataset(_ClassificationBase):

    def __init__(
            self,
            root_dir: str,
            image_scale: Optional[float] = None,
            image_size: Optional[tuple[int, int]] = None,
            preserve_aspect_ratio: bool = True,
            interpolation: Optional[int] = None,
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
        image_scale : float, optional
            Scale factor to resize images. Default is None (no scaling).
        image_size : tuple, optional
            Size of the images to be resized to (height, width). Default is None.
        preserve_aspect_ratio : bool, optional
            If True, preserve the aspect ratio of the images when resizing. Default is True.
        interpolation : int, optional
            Interpolation method to use when resizing images. Default is None (uses default interpolation).

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
        super().__init__(
            image_scale=image_scale,
            image_size=image_size,
            preserve_aspect_ratio=preserve_aspect_ratio,
            interpolation=interpolation
        )

        self.root_dir = Path(root_dir)

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
        
        img_path = self.root_dir / label / self.filenames[index]
        image = imread(img_path, mode="RGB")
        image = self._preprocess_image(image)

        return image, self.class_name_to_index(label)
