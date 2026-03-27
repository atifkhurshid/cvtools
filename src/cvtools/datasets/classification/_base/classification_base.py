"""
Base class for classification datasets.
"""

# Author: Atif Khurshid
# Created: 2025-06-18
# Modified: 2026-03-27
# Version: 1.4
# Changelog:
#     - 2026-03-03: Added _preprocess_image method to handle image scaling and resizing in a consistent way across datasets.
#     - 2026-03-26: Merged repeated code into base class.
#     - 2026-03-27: Refactored base class into separate base classes for image-based and HDF5-based datasets.

from typing import Optional, Union, Callable

import cv2
import numpy as np

from ....image import imscale, imresize, imresize_maximum


class _ClassificationBase:
    """
    Base class for classification datasets.
    This class provides a common interface for classification datasets.
    """

    def __init__(
            self,
            root_dir: str,
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
        classes : list[str]
            List of class names.
        class2index : dict[str, int]
            Mapping from class names to integer indices.
        index2class : dict[int, str]
            Mapping from integer indices to class names.
        labels : list[str]
            List of labels corresponding to the images.
        """
        self.classes: list
        self.labels: list
        self.class2index: dict
        self.index2class: dict
        self._read_image: Callable[[str], np.ndarray]

        self.root_dir = root_dir
        self.image_scale = image_scale
        self.image_size = image_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.interpolation = interpolation if interpolation is not None else cv2.INTER_AREA


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
            
            image = imscale(
                image,
                self.image_scale,
                interpolation=self.interpolation
            )

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
    

    def _initialize(self):
        """
        Initialize the dataset by setting up class names, labels, and mappings.
        This method should be called at the end of the subclass' constructor.
        """
        self.class2index = {c: i for i, c in enumerate(self.classes)}
        self.index2class = {i: c for c, i in self.class2index.items()}


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
