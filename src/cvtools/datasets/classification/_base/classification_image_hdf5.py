"""
Base class for classification datasets sourced from both image and HDF5 files.
"""

# Author: Atif Khurshid
# Created: 2026-04-08
# Modified: 2026-04-17
# Version: 1.0
# Changelog:
#     - 2026-04-08: Created base class for datasets sourced from both image and HDF5 files.
#     - 2026-04-17: Aligned with updated base class.

from typing import Optional, Union

from .classification_image import _ClassificationBaseImage
from .classification_hdf5 import _ClassificationBaseHDF5


class _ClassificationBaseImageHDF5(_ClassificationBaseImage, _ClassificationBaseHDF5):
    """
    Base class for classification datasets sourced from both image and HDF5 files.
    """

    def __init__(
            self,
            root_dir: str,
            hdf5_mode: Optional[str] = None,
            hdf5_path: Optional[str] = "images.hdf5",
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
        hdf5_mode : str, optional
            If "stream", load images from an HDF5 file on-the-fly.
            If "preload", preload all images from the HDF5 file into memory. Default is None (load from files).
        hdf5_path : str, optional
            Path to the HDF5 file within the root directory. Default is "images.hdf5".
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

        if hdf5_mode:

            self._parent_class = _ClassificationBaseHDF5

            _ClassificationBaseHDF5.__init__(
                self,
                root_dir=root_dir,
                hdf5_mode=hdf5_mode,
                hdf5_path=hdf5_path,
                image_scale=image_scale,
                image_size=image_size,
                preserve_aspect_ratio=preserve_aspect_ratio,
                interpolation=interpolation
            )

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


    def __getattr__(self, name):

        return getattr(self._parent_class, name).__get__(self)
