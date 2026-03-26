"""
Dataloader for NIH Chest X-Ray dataset: https://nihcc.app.box.com/v/ChestXray-NIHCC
"""

# Author: Atif Khurshid
# Created: 2025-05-20
# Modified: 2026-03-26
# Version: 2.4
# Changelog:
#     - 2025-05-22: Add image_size parameter for resizing images
#     - 2025-05-22: Remove pytorch dependency and refactor code
#     - 2025-05-23: Add translation between class names and labels
#     - 2025-05-29: Add labels as an attribute
#     - 2026-02-12: Add option to specify view position (AP/PA)
#     - 2026-02-13: Add option to specify class mode (binary/singleclass/multiclass)
#     - 2026-03-03: Refactored code to use new image processing functions.
#     - 2026-03-26: Refactored code to match updated base class.

import os
from typing import Optional

import pandas as pd

from .._base import _ClassificationBase


class CXRDataset(_ClassificationBase):
    def __init__(
        self,
        root_dir: str,
        view: str = "AP",
        train: bool = True,
        class_mode: str = "singleclass",
        image_mode: str = "GRAY",
        image_scale: Optional[float] = None,
        image_size: Optional[tuple[int, int]] = None,
        preserve_aspect_ratio: bool = True,
        interpolation: Optional[int] = None,
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
        view : str, optional
            View position of the chest X-ray images to load.
            Can be "AP" (Anterior-Posterior), "PA" (Posterior-Anterior), or both.
            Default is "AP".
        train : bool, optional
            If True, load training/validation data. If False, load test data. Default is True.
        class_mode : str, optional
            Mode for class labels. Can be "binary" (0 for 'No Finding', 1 for 'Finding'),
            "singleclass" (only the first label for samples with multiple labels),
            or "multiclass" (all labels as they are). Default is "singleclass".
        image_mode : str, optional
            Mode to read images. Default is "GRAY" for grayscale images.
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
        >>> dataset = CXRDataset(root_dir='/path/to/dataset', image_size=(224, 224), train=True, class_mode="binary")
        >>> print(len(dataset))  # Number of samples in the dataset
        >>> image, label = dataset[0]
        >>> print(image.shape, label)
        >>> for image, label in dataset:
        ...     # Process each image and label
        ...     pass
        """
        super().__init__(
            root_dir=root_dir,
            image_mode=image_mode,
            image_scale=image_scale,
            image_size=image_size,
            preserve_aspect_ratio=preserve_aspect_ratio,
            interpolation=interpolation
        )

        self.images_dir = os.path.join(self.root_dir, 'images')
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Directory {self.images_dir} does not exist.")
        
        # Load annotations file
        self.data = pd.read_csv(os.path.join(self.root_dir, 'Data_Entry_2017_v2020.csv'))

        # Filter data based on view position
        assert view in ["AP", "PA", "both"], \
            f"Invalid view position: {view}. Must be 'AP', 'PA', or 'both'."
        if view == "AP":
            self.data = self.data[self.data["View Position"] == "AP"]
        elif view == "PA":
            self.data = self.data[self.data["View Position"] == "PA"]

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

        assert class_mode in ["binary", "singleclass", "multiclass"], \
            f"Invalid class_mode: {class_mode}. Must be 'binary', 'singleclass', or 'multiclass'."
        
        if class_mode == "binary":
            # Convert the multiclass textual labels to binary
            # 0 for 'No Finding' and 1 for 'Finding'
            self.data['Finding Labels'] = self.data['Finding Labels'].apply(
                lambda x: "Normal" if x == 'No Finding' else "Abnormal")
        elif class_mode == "singleclass":
            # For samples with multiple labels, take only the first label as the class label
            self.data['Finding Labels'] = self.data['Finding Labels'].str.split('|').str[0]

        self.labels = self.data['Finding Labels'].tolist()
        self.classes = sorted(self.data['Finding Labels'].unique().tolist())

        self.__initialize__()


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
        image_path = os.path.join(
            self.images_dir,
            str(self.data.loc[index, 'Image Index'])
        )
        label = self.labels[index]

        return image_path, label
