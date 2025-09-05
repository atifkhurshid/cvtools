"""
Dataloader for 8-category Scenes dataset from Oliva & Torralba (2001)
Link: https://people.csail.mit.edu/torralba/code/spatialenvelope/
"""

# Author: Atif Khurshid
# Created: 2025-09-03
# Modified: None
# Version: 1.0
# Changelog:
#     - None

import os

import numpy as np

from .._base import _ClassificationBase
from ....image import imread



class Scenes8Dataset(_ClassificationBase):
    def __init__(
            self,
            root_dir: str,
            image_size: tuple[int, int] | None = None,
            preserve_aspect_ratio: bool = True,
        ):
        """
        8-category Scenes dataset loader.

        This class loads images and labels from the 8-category Scenes dataset.
        The dataset is expected to be organized with all images in
        in a single "images" folder within the root directory and
        filenames in the format "classname_*.jpg".

        Parameters
        ----------
        root_dir : str
            Path to the root directory of the dataset.
        image_size : tuple, optional
            Size of the images to be resized to (height, width). Default is None.
        preserve_aspect_ratio : bool, optional
            If True, preserve the aspect ratio of the images when resizing. Default is True.

        Attributes
        ----------
        images_dir : str
            Path to the directory containing the images.
        filenames : list
            List of filenames of the images.
        labels : list
            List of class labels corresponding to the images.
        classes : list
            List of unique class labels in the dataset.

        Examples
        --------
        >>> from cvtools.datasets import Scenes8Dataset
        >>> dataset = Scenes8Dataset(root_dir='path/to/scenes', image_size=(224, 224))
        >>> for img, label in dataset:
        ...     # Process each image and label
        ...     pass
        """
        self.root_dir = root_dir
        self.image_size = image_size    # (height, width)
        self.preserve_aspect_ratio = preserve_aspect_ratio

        self.images_dir = os.path.join(self.root_dir, 'images')
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Directory {self.images_dir} does not exist.")

        self.filenames = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        self.labels = [f.split("_")[0] for f in self.filenames]
        self.classes = sorted(list(set(self.labels)))

        self.__initialize__()


    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """
        Returns the image and label at the specified index.
        The image is read in RGB format and resized to the specified image size.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple[np.ndarray, int]
            A tuple containing the image as a numpy array and the label index.
        """
        # Read filepath from the list
        img_path = os.path.join(self.images_dir, self.filenames[idx])
        # Read image as RGB
        image = imread(
            img_path,
            mode="RGB",
            size=self.image_size,
            preserve_aspect_ratio=self.preserve_aspect_ratio,
        )
        # Read label from the list and convert to label index
        label = self.class_name_to_index(self.labels[idx])

        return image, label
