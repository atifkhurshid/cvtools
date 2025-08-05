"""
Dataloader for Princeton SUN dataset: https://vision.princeton.edu/projects/2010/SUN/
"""

# Author: Atif Khurshid
# Created: 2025-05-23
# Modified: None
# Version: 1.0
# Changelog:
#     - None

import os

import numpy as np
import pandas as pd

from .._base import _ClassificationBase
from ....image import imread


class SUNDataset(_ClassificationBase):
    def __init__(
            self,
            root_dir: str,
            class_hierarchy: str="basic",
            image_size: tuple[int, int] | None=None,
            preserve_aspect_ratio: bool=True,
            train: bool=True
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
        image_size : tuple, optional
            Size of the images to be resized to (height, width). Default is None.
        preserve_aspect_ratio : bool, optional
            If True, preserve the aspect ratio of the images when resizing. Default is True.
        train : bool, optional
            If True, load training/validation data. If False, load test data. Default is True.
        
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
        self.root_dir = root_dir
        self.image_size = image_size    # (height, width)
        self.preserve_aspect_ratio = preserve_aspect_ratio

        self.images_dir = os.path.join(self.root_dir, 'images')
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Directory {self.images_dir} does not exist.")
    
        if train:
            filepaths_list_path = os.path.join(self.root_dir, 'metadata', 'Training_01.txt')
        else:
            filepaths_list_path = os.path.join(self.root_dir, 'metadata', 'Testing_01.txt')

        # Read list of train images
        with open(filepaths_list_path, 'r') as f:
            filepaths_list = f.read().split('\n')[:-1]    # Remove last empty row

        # Infer class from filepath
        # Example: /a/airport/entrance/abckjaskasd.jpg
        # Class is the 2nd (or possibly including 3rd part) of the filepath
        self.filepaths = []
        self.labels = []
        for filepath in filepaths_list:
            parts = filepath.split("/")
            if len(parts) == 4:
                self.labels.append(parts[2])
            elif len(parts) == 5:
                self.labels.append(f"{parts[2]}/{parts[3]}")
            else:
                print(f"WARNING: Filepath {filepath} will be skipped because class could not be inferred.")
                continue
            self.filepaths.append(filepath[1:])    # Remove leading '/'

        assert len(self.filepaths) == len(self.labels), "Number of filepaths and classes do not match."

        if class_hierarchy != "sun":
            # Read class hierarchy from CSV file
            class_hierarchy_df = pd.read_csv(os.path.join(self.root_dir, 'metadata', 'class_hierarchy.csv'))
            class_hierarchy_df = class_hierarchy_df.set_index('class')
            # Change label name according to the class hierarchy
            self.labels = [class_hierarchy_df.loc[x, class_hierarchy] for x in self.labels]

        self.classes = sorted(list(set(self.labels)))

        self.__initialize__()


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
        # Read filepath from the list
        img_path = os.path.join(self.images_dir, self.filepaths[idx])
        # Read image as RGB
        image = imread(
            img_path,
            mode="RGB",
            size=self.image_size,
            preserve_aspect_ratio=self.preserve_aspect_ratio,
        )
        # Read label from the list and convert to label index
        label = self.class_name_to_index(self.labels[idx])

        return image, label
