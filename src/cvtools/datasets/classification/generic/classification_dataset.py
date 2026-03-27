"""
Generic dataloader for image classification tasks.
"""

# Author: Atif Khurshid
# Created: 2022-12-18
# Modified: 2026-03-27
# Version: 1.3
# Changelog:
#     - 2025-06-18: Updated documentation and type hints.
#     - 2025-10-30: Refactored image loading to use imread function.
#     - 2026-03-03: Refactored code to use new image processing functions.
#     - 2026-03-26: Refactored code to match updated base class.
#     - 2026-03-27: Refactored code to match updated base class.

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .._base import _ClassificationBaseImage, _ClassificationBaseHDF5
from ....image import IMAGE_EXTENSIONS


class ClassificationDataset(_ClassificationBaseImage, _ClassificationBaseHDF5):

    def __init__(
            self,
            root_dir: str,
            exts: Union[list[str], str] = "auto",
            hdf5_mode: Optional[str] = None,
            image_mode: Union[str, int] = 'RGB',
            image_scale: Optional[float] = None,
            image_size: Optional[Union[int, tuple[int, int]]] = None,
            preserve_aspect_ratio: bool = True,
            interpolation: Optional[int] = None,
        ):
        """
        Generic classification dataset class.

        This class loads images from a directory structure where each subdirectory
        corresponds to a class label.

        Parameters
        ---------- 
        root_dir : str
            Path to the root directory containing class subdirectories.
        exts : list[str] | str, optional
            List of file extensions to consider as valid images. Default is "auto" (uses IMAGE_EXTENSIONS).
        hdf5_mode : str, optional
            If "stream", load images from an HDF5 file on-the-fly.
            If "preload", preload all images from the HDF5 file into memory. Default is None (load from files).
        image_mode : str | int, optional
            Mode to read images. Can be 'RGB', 'GRAY', or a cv2.IMREAD_... flag. Default is 'RGB'.
        image_scale : float, optional
            Scale factor to resize images. Default is None (no scaling).
        image_size : int | tuple, optional
            Size of the images to be resized to. If int, resizes the maximum dimension to this size.
            If tuple, should be (height, width). Default is None (no resizing).
        preserve_aspect_ratio : bool, optional
            If True, preserve the aspect ratio of the images when resizing. Default is True.
        interpolation : int, optional
            Interpolation method to use when resizing images. Default is None (uses default interpolation).

        Attributes
        ----------
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
        if exts == "auto":
            exts = IMAGE_EXTENSIONS

        if hdf5_mode is not None:

            self._parent_class = _ClassificationBaseHDF5

            _ClassificationBaseHDF5.__init__(
                self,
                root_dir=root_dir,
                hdf5_mode=hdf5_mode,
                image_scale=image_scale,
                image_size=image_size,
                preserve_aspect_ratio=preserve_aspect_ratio,
                interpolation=interpolation
            )

            self.root_dir = ""
            
            self.classes = list(self.images_file.keys())

            self.labels = []
            self.filenames = []
            for i, c in enumerate(self.classes):
                filenames = list(self.images_file[c].keys())
                self.labels.extend([c] * len(filenames))
                self.filenames.extend(filenames)

        else:

            self._parent_class = _ClassificationBaseImage

            _ClassificationBaseImage.__init__(
                self,
                root_dir=root_dir,
                image_mode=image_mode,
                image_scale=image_scale,
                image_size=image_size,
                preserve_aspect_ratio=preserve_aspect_ratio,
                interpolation=interpolation
            )

            root_dir_path = Path(self.root_dir)

            self.classes = [x.name for x in root_dir_path.iterdir() if not x.is_file()]

            self.labels = []
            self.filenames = []
            for i, c in enumerate(self.classes):
                directory = root_dir_path / c
                filenames = [x.name for x in directory.iterdir() if x.suffix in exts]
                labels = [c] * len(filenames)
                self.filenames.extend(filenames)
                self.labels.extend(labels)

        self.ids = np.arange(len(self.filenames))

        self._initialize()


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
        id = self.ids[index]
        label = self.labels[id]
        image_path = os.path.join(self.root_dir, label, self.filenames[id])

        return image_path, label


    def __getattr__(self, name):
        return getattr(self._parent_class, name).__get__(self)
    