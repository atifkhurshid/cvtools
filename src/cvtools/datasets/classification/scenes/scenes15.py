"""
Dataloader for 15-category Scenes dataset from Lazebnik et al. (2006)
Link: https://ieeexplore.ieee.org/document/1641019
"""

# Author: Atif Khurshid
# Created: 2022-09-05
# Modified: 2026-03-26
# Version: 1.3
# Changelog:
#     - 2026:02-10: Used custom imread function.
#     - 2026-03-03: Refactored code to use new image processing functions.
#     - 2026-03-26: Refactored code to match updated base class.

from pathlib import Path
from typing import Optional

from .._base import _ClassificationBase


class Scenes15Dataset(_ClassificationBase):

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
        15-category Scenes dataset loader.

        This class loads images and labels from the 15-category Scenes dataset.
        The dataset is expected to be organized with all images belonging to each class
        in separate subdirectories within the root directory.

        Parameters
        ---------- 
        root_dir : str
            Path to the root directory containing class subdirectories.
        image_mode : str, optional
            Color mode for loading images (e.g., 'RGB', 'L'). Default is 'RGB'.
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
            root_dir=root_dir,
            image_mode=image_mode,
            image_scale=image_scale,
            image_size=image_size,
            preserve_aspect_ratio=preserve_aspect_ratio,
            interpolation=interpolation
        )

        self.root_dir = Path(self.root_dir)

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
        label = self.labels[index]
        image_path = str(self.root_dir / label / self.filenames[index])

        return image_path, label
