"""
Dataloader for Princeton SUN dataset: https://vision.princeton.edu/projects/2010/SUN/
"""

# Author: Atif Khurshid
# Created: 2025-05-23
# Modified: 2026-03-03
# Version: 1.2
# Changelog:
#     - 2026-03-03: Added option to load images from HDF5 file for faster loading.
#     - 2026-03-03: Enabled dynamic changes in class hierarchy after initialization.
#     - 2026-03-03: Refactored code to remove redundant pytorch dataset classes.
#     - 2026-03-03: Added support for image scaling.

import os
from typing import Optional

import h5py
import numpy as np
import pandas as pd

from .._base import _ClassificationBase
from ....image import imread
from ....utils import stratified_sampling_by_class


class SUNDataset(_ClassificationBase):
    def __init__(
            self,
            root_dir: str,
            class_hierarchy: str = "basic",
            train: bool = True,
            split_idx: int = 0,
            n_samples: int = 0,
            hdf5_mode: bool = False,
            image_scale: Optional[float] = None,
            image_size: Optional[tuple[int, int]] = None,
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
        hdf5_mode : bool, optional
            If True, load images from an HDF5 file instead of individual image files. Default is False.
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
            image_scale=image_scale,
            image_size=image_size,
            preserve_aspect_ratio=preserve_aspect_ratio,
            interpolation=interpolation
        )

        self.root_dir = root_dir
        self.hdf5_mode = hdf5_mode

        if hdf5_mode:
            self.images_file = h5py.File(os.path.join(self.root_dir, "images.hdf5"), "r")
            self._read_image = self._read_image_from_hdf5
        else:
            self.images_dir = os.path.join(self.root_dir, 'images')
            if not os.path.exists(self.images_dir):
                raise FileNotFoundError(f"Directory {self.images_dir} does not exist.")
            self._read_image = self._read_image_from_file
    
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


    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """
        Returns the image and label at the specified index.
        The image is read in RGB format and resized to the specified image size.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple[np.ndarray, int]
            A tuple containing the image as a numpy array and the label index.
        """
        image = self._read_image(idx)
        image = self._preprocess_image(image)
        label = self.class_name_to_index(self.labels[idx])

        return image, label


    def _read_image_from_file(self, idx: int) -> np.ndarray:
        """
        Reads an image from the file system based on the index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        np.ndarray
            The image as a numpy array.
        """
        img_path = os.path.join(self.images_dir, self.filepaths[idx])
        image = imread(img_path, mode="RGB")
        
        return image


    def _read_image_from_hdf5(self, idx: int) -> np.ndarray:
        """
        Reads an image from the HDF5 file based on the index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        np.ndarray
            The image as a numpy array.
        """
        img_path = self.filepaths[idx]
        image = self.images_file['/' + img_path][:]
            
        return image
    

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
        self.__initialize__()


    def __del__(self):
        """
        Closes the HDF5 file if it was opened in hdf5_mode when the dataset object is deleted.
        """
        if self.hdf5_mode and self.images_file is not None:
            self.images_file.close()
            self.images_file = None
