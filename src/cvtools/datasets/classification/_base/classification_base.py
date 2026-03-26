"""
Base class for classification datasets.
"""

# Author: Atif Khurshid
# Created: 2025-06-18
# Modified: 2026-03-26
# Version: 1.3
# Changelog:
#     - 2026-03-03: Added _preprocess_image method to handle image scaling and resizing in a consistent way across datasets.
#     - 2026-03-26: Merged repeated code into base class.

from typing import Optional, Union

import os
import cv2
import h5py
import numpy as np

from ....image import imread, imscale, imresize, imresize_maximum


class _ClassificationBase:
    """
    Base class for classification datasets.
    This class provides a common interface for classification datasets.
    """

    def __init__(
            self,
            root_dir: str,
            image_mode: Union[str, int] = 'RGB',
            hdf5_mode: bool = False,
            image_scale: Optional[float] = None,
            image_size: Optional[Union[int, tuple[int, int]]] = None,
            preserve_aspect_ratio: bool = True,
            interpolation: Optional[int] = None,
        ):

        self.classes: list
        self.labels: list
        self.class2index: dict
        self.index2class: dict
        self.images_file: h5py.File

        self.root_dir = root_dir
        self.image_mode = image_mode
        self.hdf5_mode = hdf5_mode
        self.image_scale = image_scale
        self.image_size = image_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.interpolation = interpolation if interpolation is not None else cv2.INTER_AREA

        if self.hdf5_mode:
            self._read_image = self._read_image_from_hdf5
            self.images_file = h5py.File(os.path.join(self.root_dir, "images.hdf5"), "r")
        else:
            self._read_image = self._read_image_from_file


    def __initialize__(self):
        """
        Initialize the dataset by setting up class names, labels, and mappings.
        This method should be called at the end of the subclass' constructor.
        """
        self.class2index = {c: i for i, c in enumerate(self.classes)}
        self.index2class = {i: c for c, i in self.class2index.items()}


    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.labels)


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
        image_path, label = self._get_image_path_and_label(index)

        image = self._read_image(image_path)
        image = self._preprocess_image(image)

        label = self.class_name_to_index(label)

        return image, label


    def __del__(self):
        """
        Closes the HDF5 file if it was opened in hdf5_mode when the dataset object is deleted.
        """
        if self.hdf5_mode and self.images_file is not None:
            self.images_file.close()
            self.images_file = None


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
        msg = "Subclasses must implement the _get_image_path_and_label method to return" \
              "the image path and label for a given index."
        raise NotImplementedError(msg)


    @property
    def num_classes(self) -> int:
        """
        Return the number of classes in the dataset.
        
        Returns
        -------
        int
            The number of classes.
        """
        return len(self.classes)
    

    def class_name_to_index(self, x: str) -> int:
        """
        Convert a class name to its corresponding index.

        Parameters
        ----------
        x : str
            The class name to convert.

        Returns
        -------
        int
            The corresponding index for the class name.
        """
        return self.class2index.get(x, -1)


    def index_to_class_name(self, x: int) -> str:
        """
        Convert an index to its corresponding class name.

        Parameters
        ----------
        x : int
            The index to convert.

        Returns
        -------
        str
            The corresponding class name for the index.
        """
        return self.index2class.get(x, "Unknown")


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


    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process the image by applying scaling and resizing if specified.

        Parameters
        ----------
        image : np.ndarray
            The input image to process.

        Returns
        -------
        np.ndarray
            The processed image.
        """
        if self.image_scale is not None:
            image = imscale(image, self.image_scale, interpolation=self.interpolation)

        if self.image_size is not None:
            if isinstance(self.image_size, int):
                image = imresize_maximum(
                    image,
                    max_size=self.image_size,
                    interpolation=self.interpolation
                )
            else:
                image = imresize(
                    image,
                    size=self.image_size,
                    preserve_aspect_ratio=self.preserve_aspect_ratio,
                    interpolation=self.interpolation
                )

        return image
