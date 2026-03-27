"""
Base class for classification datasets sourced from image files.
"""

# Author: Atif Khurshid
# Created: 2026-03-27
# Modified: None
# Version: 1.0
# Changelog:
#     - 2026-03-27: Refactored base class into separate base classes for image-based and HDF5-based datasets.

from typing import Optional, Union

import numpy as np

from .classification_base import _ClassificationBase
from ....image import imread


class _ClassificationBaseImage(_ClassificationBase):
    """
    Base class for classification datasets sourced from image files.
    """

    def __init__(
            self,
            root_dir: str,
            image_mode: Union[str, int] = 'RGB',
            image_scale: Optional[float] = None,
            image_size: Optional[Union[int, tuple[int, int]]] = None,
            preserve_aspect_ratio: bool = True,
            interpolation: Optional[int] = None,
        ):
        """
        Initializes the classification dataset.

        Parameters
        ----------
        root_dir : str
            Path to the root directory containing class subdirectories or HDF5 file.
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
            Interpolation method to use when resizing images. Should be a cv2.INTER_... flag.
            Default is None.
        """
        super().__init__(
            root_dir=root_dir,
            image_scale=image_scale,
            image_size=image_size,
            preserve_aspect_ratio=preserve_aspect_ratio,
            interpolation=interpolation,
        )
        self.image_mode = image_mode
        self._read_image = self._read_image_from_file


    def _read_image_from_file(self, path: str) -> np.ndarray:
        """
        Reads an image from the file system based on the index.

        Parameters
        ----------
        path : str
            Path to the image file.

        Returns
        -------
        np.ndarray
            The image as a numpy array.
        """
        image = imread(path, mode=self.image_mode)
        
        return image
