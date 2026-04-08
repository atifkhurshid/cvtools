"""
Dataloader for Princeton SUN dataset: https://vision.princeton.edu/projects/2010/SUN/
"""

# Author: Atif Khurshid
# Created: 2025-05-23
# Modified: 2026-04-08
# Version: 1.3
# Changelog:
#     - 2026-03-03: Added option to load images from HDF5 file for faster loading.
#     - 2026-03-03: Enabled dynamic changes in class hierarchy after initialization.
#     - 2026-03-03: Refactored code to remove redundant pytorch dataset classes.
#     - 2026-03-03: Added support for image scaling.
#     - 2026-03-26: Refactored code to match updated base class.
#     - 2026-03-27: Refactored code to match updated base class.
#     - 2026-04-08: Refactored code to match updated base class.

import os
from typing import Optional, Union

import numpy as np
import pandas as pd

from .._base import _ClassificationBaseImageHDF5
from ....utils import stratified_sampling_by_class


class SUNDataset(_ClassificationBaseImageHDF5):
    def __init__(
            self,
            root_dir: str,
            class_hierarchy: str = "basic",
            train: bool = True,
            split_idx: int = 0,
            n_samples: int = 0,
            hdf5_mode: Optional[str] = None,
            image_mode: str = 'RGB',
            image_scale: Optional[float] = None,
            image_size: Optional[Union[int, tuple[int, int]]] = None,
            preserve_aspect_ratio: bool = True,
            interpolation: Optional[int] = None,
        ):
        """
        Princeton SUN dataset loader.

        This class loads images and labels from the Princeton SUN dataset.
        The dataset is expected to be organized in a specific directory structure
        and the annotations are provided in text files.

        Parameters
        ----------
        root_dir : str
            Path to the root directory of the dataset.
        class_hierarchy : str, optional
            Class hierarchy to use. Options are "sun", "basic", "superordinate", or "binary". Default is "basic".
        train : bool, optional
            If True, load training/validation data. If False, load test data. Default is True.
        split_idx : int, optional
            Index of the split to use. The dataset is divided into 10 splits. Default is 0.
        n_samples : int, optional
            Number of samples to load from each class. This is used for stratified sampling. Default is 0 (no sampling).
        image_mode : str, optional
            Mode to read images. Can be 'RGB', 'GRAY', or a cv2.IMREAD_... flag. Default is 'RGB'.
        hdf5_mode : str, optional
            If "stream", load images from an HDF5 file on-the-fly.
            If "preload", preload all images from the HDF5 file into memory. Default is None (load from files).
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
        images_dir : str
            Path to the directory containing the images.
        filepaths : list
            List of filepaths to the images.
        labels : list
            List of class labels corresponding to the images.
        classes : list
            List of unique class labels in the dataset.
        label2index : dict
            Mapping from class labels to indices.
        index2label : dict
            Mapping from indices to class labels.

        Examples
        --------
        >>> dataset = SUNDataset(root_dir='/path/to/sun', class_hierarchy='basic', image_size=(224, 224), train=True)
        >>> print(len(dataset))  # Number of samples in the dataset
        >>> image, label = dataset[0]
        >>> print(image.shape, label)
        >>> for image, label in dataset:
        ...     # Process each image and label
        ...     pass

        """
        super().__init__(
            root_dir=root_dir,
            hdf5_mode=hdf5_mode,
            image_mode=image_mode,
            image_scale=image_scale,
            image_size=image_size,
            preserve_aspect_ratio=preserve_aspect_ratio,
            interpolation=interpolation,
        )

        if hdf5_mode:
            self.images_dir = ""

        else:
            self.images_dir = os.path.join(self.root_dir, "images")

            if not os.path.exists(self.images_dir):
                raise FileNotFoundError(f"Directory {self.images_dir} does not exist.")
    
        splits = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
        assert split_idx < len(splits), f"Invalid split index {split_idx}. Must be less than {len(splits)}."

        if train:
            filepaths_list_path = os.path.join(self.root_dir, 'metadata', f'Training_{splits[split_idx]}.txt')
        
        else:
            filepaths_list_path = os.path.join(self.root_dir, 'metadata', f'Testing_{splits[split_idx]}.txt')

        # Read list of train images
        with open(filepaths_list_path, 'r') as f:
            
            filepaths_list = f.read().split('\n')[:-1]    # Remove last empty row

        # Infer class from filepath
        # Example: /a/airport/entrance/abckjaskasd.jpg
        # Class is the 2nd (or possibly including 3rd part) of the filepath
        self.filepaths = []
        self.sun_labels = []

        for filepath in filepaths_list:

            parts = filepath.split("/")

            if len(parts) == 4:
                self.sun_labels.append(parts[2])

            elif len(parts) == 5:
                self.sun_labels.append(f"{parts[2]}/{parts[3]}")

            else:
                print(f"WARNING: Filepath {filepath} will be skipped because class could not be inferred.")
                continue

            self.filepaths.append(filepath[1:])    # Remove leading '/'

        assert len(self.filepaths) == len(self.sun_labels), "Number of filepaths and classes do not match."

        # Perform stratified sampling if n_samples is specified
        if n_samples > 0:

            self.filepaths, self.sun_labels = stratified_sampling_by_class(
                np.array(self.filepaths),
                np.array(self.sun_labels),
                n_samples=n_samples,
                seed=42
            )
            self.filepaths = self.filepaths.tolist()
            self.sun_labels = self.sun_labels.tolist()

        # Read class hierarchy from CSV file
        class_hierarchy_df = pd.read_csv(os.path.join(self.root_dir, 'metadata', 'class_hierarchy.csv'))
        class_hierarchy_df = class_hierarchy_df.set_index('class')
        self.labels_dict = {
            "sun": self.sun_labels,
            "basic": [class_hierarchy_df.loc[x, "basic"] for x in self.sun_labels],
            "superordinate": [class_hierarchy_df.loc[x, "superordinate"] for x in self.sun_labels],
            "binary": [class_hierarchy_df.loc[x, "binary"] for x in self.sun_labels],
        }

        self.set_class_hierarchy(class_hierarchy)


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
        image_path = os.path.join(self.images_dir, self.filepaths[index])
        label = self.labels[index]

        return image_path, label


    def set_class_hierarchy(self, class_hierarchy: str):
        """
        Sets the class hierarchy for the dataset.

        Parameters
        ----------
        class_hierarchy : str
            Class hierarchy to use. Options are "sun", "basic", "superordinate", or "binary".

        Raises
        ------
        ValueError
            If an invalid class hierarchy is specified.
        """
        if class_hierarchy not in self.labels_dict:
            raise ValueError(f"Invalid class hierarchy {class_hierarchy}.\
                             Must be one of {list(self.labels_dict.keys())}.")
        
        self.class_hierarchy = class_hierarchy
        self.labels = self.labels_dict[class_hierarchy]
        self.classes = sorted(list(set(self.labels)))
        self._initialize()
