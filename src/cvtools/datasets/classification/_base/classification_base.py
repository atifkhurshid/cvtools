"""
Base class for classification datasets.
"""

# Author: Atif Khurshid
# Created: 2025-06-18
# Modified: 2026-03-03
# Version: 1.2
# Changelog:
#     - 2026-03-03: Added _preprocess_image method to handle image scaling and resizing in a consistent way across datasets.

from typing import Optional, Union

import cv2

from ....image import imscale, imresize, imresize_maximum


class _ClassificationBase:
    """
    Base class for classification datasets.
    This class provides a common interface for classification datasets.
    """

    def __init__(
            self,
            image_scale: Optional[float] = None,
            image_size: Optional[Union[int, tuple[int, int]]] = None,
            preserve_aspect_ratio: bool = True,
            interpolation: Optional[int] = None,
        ):

        self.classes: list
        self.labels: list
        self.class2index: dict
        self.index2class: dict

        self.image_scale = image_scale
        self.image_size = image_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.interpolation = interpolation if interpolation is not None else cv2.INTER_AREA


    def __initialize__(self):
        """
        Initialize the dataset by setting up class names, labels, and mappings.
        This method should be called at the end of the subclass' constructor.
        """
        self.class2index = {cls: idx for idx, cls in enumerate(self.classes)}
        self.index2class = {idx: cls for cls, idx in self.class2index.items()}


    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.labels)


    def __getitem__(self, index: int) -> tuple:
        """
        Get a sample from the dataset.
        
        Parameters
        ----------
        index : int
            Index of the sample to retrieve.
        
        Returns
        -------
        tuple
            A tuple containing the sample data and its label.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    

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


    def _preprocess_image(self, image):
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
    