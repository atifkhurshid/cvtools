"""
Dataloader for 8-category Scenes dataset from Oliva & Torralba (2001)
Link: https://people.csail.mit.edu/torralba/code/spatialenvelope/
"""

# Author: Atif Khurshid
# Created: 2025-09-03
# Modified: 2026-03-26
# Version: 1.2
# Changelog:
#     - 2026-03-03: Refactored code to use new image processing functions.
#     - 2026-03-26: Refactored code to match updated base class.

import os
from typing import Optional

from .._base import _ClassificationBase


class Scenes8Dataset(_ClassificationBase):
    def __init__(
            self,
            root_dir: str,
            image_mode: str = 'RGB',
            image_scale: Optional[float] = None,
            image_size: Optional[tuple[int, int]] = None,
            preserve_aspect_ratio: bool = True,
            interpolation: Optional[int] = None,
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
        image_mode : str, optional
            Mode to read images. Can be 'RGB', 'GRAY', or a cv2.IMREAD_... flag. Default is 'RGB'.
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
        super().__init__(
            root_dir=root_dir,
            image_mode=image_mode,
            image_scale=image_scale,
            image_size=image_size,
            preserve_aspect_ratio=preserve_aspect_ratio,
            interpolation=interpolation
        )

        self.images_dir = os.path.join(self.root_dir, 'images')
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Directory {self.images_dir} does not exist.")

        self.filenames = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        self.labels = [f.split("_")[0] for f in self.filenames]
        self.classes = sorted(list(set(self.labels)))

        self.__initialize__()


    def _get_image_path_and_label(self, index: int) -> tuple[str, str]:
        """
        Get the image path and label for a given index.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        tuple[str, str]
            A tuple containing the image path and its corresponding label.

        """
        image_path = os.path.join(self.images_dir, self.filenames[index])
        label = self.labels[index]

        return image_path, label
