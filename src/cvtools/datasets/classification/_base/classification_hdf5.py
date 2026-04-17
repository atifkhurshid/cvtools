"""
Base class for classification datasets sourced from HDF5 file.
"""

# Author: Atif Khurshid
# Created: 2026-03-27
# Modified: 2026-04-17
# Version: 1.1
# Changelog:
#     - 2026-03-27: Refactored base class into separate base classes for image-based and HDF5-based datasets.
#     - 2026-04-17: Allowed custom HDF5 file paths

from typing import Optional, Union

import os
import h5py
import numpy as np

from .classification_base import _ClassificationBase


class _ClassificationBaseHDF5(_ClassificationBase):
    """
    Base class for classification datasets sourced from an HDF5 file.
    """

    def __init__(
            self,
            root_dir: str,
            hdf5_mode: Optional[str] = None,
            hdf5_path: Optional[str] = "images.hdf5",
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
        
        Attributes
        ----------
        images_file : h5py.File or dict
            If hdf5_mode is "stream", this will be an open h5py.File object for reading images on-the-fly.
            If hdf5_mode is "preload", this will be a dictionary mapping image paths to preloaded numpy arrays.
        """
        super().__init__(
            root_dir=root_dir,
            image_scale=image_scale,
            image_size=image_size,
            preserve_aspect_ratio=preserve_aspect_ratio,
            interpolation=interpolation,
        )

        self.hdf5_mode = hdf5_mode
        self.hdf5_path = hdf5_path
        self._read_image = self._read_image_from_hdf5

        images_file_path = os.path.join(self.root_dir, self.hdf5_path)

        if self.hdf5_mode == "stream":

            self.images_file = h5py.File(images_file_path, "r")

        elif self.hdf5_mode == "preload":

            self.images_file = {}
            with h5py.File(images_file_path, "r") as f:
                self._preload_hdf5_images(f, self.images_file)


    def _read_image_from_hdf5(self, path: str) -> np.ndarray:
        """
        Reads an image from the HDF5 file based on the index.

        Parameters
        ----------
        path : str
            Path to the image in the HDF5 file.

        Returns
        -------
        np.ndarray
            The image as a numpy array.
        """
        image = self.images_file[path][:]
        
        return image


    def _preload_hdf5_images(self, group: h5py.Group, images_dict: dict, prefix: str = ""):
        """
        Recursively preload images from an HDF5 file into a dictionary.

        Parameters
        ----------
        f : h5py.File
            The HDF5 file object to read from.
        images_dict : dict
            The dictionary to store the preloaded images.
        prefix : str, optional
            The prefix to add to the keys in the images_dict. Default is "".
        """
        for key, item in group.items():
            if prefix:
                path = f"{prefix}/{key}"
            else:
                path = key
            if isinstance(item, h5py.Dataset):
                images_dict[path] = item[:]
            elif isinstance(item, h5py.Group):
                self._preload_hdf5_images(item, images_dict, path)


    def __del__(self):
        """
        Closes the HDF5 file if it was opened in hdf5_mode when the dataset object is deleted.
        """
        if self.hdf5_mode == "stream" and self.images_file is not None:
            self.images_file.close()
            self.images_file = None
