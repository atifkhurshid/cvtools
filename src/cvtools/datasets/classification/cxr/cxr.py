"""
Dataloader for NIH Chest X-Ray dataset: https://nihcc.app.box.com/v/ChestXray-NIHCC
"""

# Author: Atif Khurshid
# Created: 2025-05-20
# Modified: 2025-09-29
# Version: 2.2
# Changelog:
#     - 2025-05-22: Add image_size parameter for resizing images
#     - 2025-05-22: Remove pytorch dependency and refactor code
#     - 2025-05-23: Add translation between class names and labels
#     - 2025-05-29: Add labels as an attribute

import os
from typing import Optional

import numpy as np
import pandas as pd

from ....image import imread
from .._base import _ClassificationBase


class CXRDataset(_ClassificationBase):
    def __init__(
            self,
            root_dir: str,
            image_size: Optional[tuple[int, int]] = None,
            preserve_aspect_ratio: bool = False,
            train: bool = True,
            binary: bool = True
    ):
        """
        NIH Chest X-Ray dataset loader.

        This class loads images and labels from the NIH Chest X-Ray dataset.
        The dataset is expected to be organized in a specific directory structure
        and the annotations are provided in a CSV file.

        Parameters
        ----------
        root_dir : str
            Path to the root directory of the dataset.
        image_size : tuple, optional
            Size of the images to be resized to (height, width). Default is None.
        train : bool, optional
            If True, load training/validation data. If False, load test data. Default is True.
        binary : bool, optional
            If True, convert labels to binary (0 for 'No Finding', 1 for 'Finding'). Default is True.
        
        Attributes
        ----------
        images_dir : str
            Path to the directory containing the images.
        data : pd.DataFrame
            DataFrame containing the annotations and labels.
        classes : list
            List of unique class labels in the dataset.
        label2idx : dict
            Mapping from class labels to indices.
        idx2label : dict
            Mapping from indices to class labels.

        Examples
        --------
        >>> dataset = CXRDataset(root_dir='/path/to/dataset', image_size=(224, 224), train=True, binary=True)
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
        
        # Load annotations file
        self.data = pd.read_csv(os.path.join(self.root_dir, 'Data_Entry_2017_v2020.csv'))

        if train:
            # Read list of train/val indices
            with open(os.path.join(self.root_dir, 'train_val_list.txt'), 'r') as f:
                train_val_list = f.read().split('\n')
            # Filter the data to include only the train/val indices
            self.data = self.data[self.data['Image Index'].isin(train_val_list)]
        else:
            # Read list of test indices
            with open(os.path.join(self.root_dir, 'test_list.txt'), 'r') as f:
                test_list = f.read().split('\n')
            # Filter the data to include only the test indices
            self.data = self.data[self.data['Image Index'].isin(test_list)]
        # Reset the index of the DataFrame to ensure it is sequential
        self.data = self.data.reset_index(drop=True)
        
        # Convert the multiclass textual labels to binary - 0 for 'No Finding' and 1 for 'Finding'
        if binary:
            self.data['Finding Labels'] = self.data['Finding Labels'].apply(
                lambda x: "Normal" if x == 'No Finding' else "Abnormal")

        self.labels = self.data['Finding Labels'].tolist()
        self.classes = sorted(self.data['Finding Labels'].unique().tolist())

        self.__initialize__()


    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        """
        Get an image and corresponding label from the dataset.
        The image is read in grayscale and resized to the specified image size.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple[np.ndarray, int]
            A tuple containing the image as a numpy array and its corresponding label index.
        """
        img_path = os.path.join(
            self.images_dir,
            str(self.data.loc[index, 'Image Index'])
        )

        image = imread(
            img_path,
            mode="GRAY",
            size=self.image_size,
            preserve_aspect_ratio=self.preserve_aspect_ratio,
        )

        label = self.class_name_to_index(self.labels[index])

        return image, label
