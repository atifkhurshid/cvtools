"""
64x64 ImageNet dataset.
"""

# Author: Atif Khurshid
# Created: 2026-03-26
# Modified: 2026-03-31
# Version: 1.1
# Changelog:
#     - 2026-03-26: Created 64x64 ImageNet dataset class.
#     - 2026-03-27: Refactored code to match updated base class.
#     - 2026-03-31: Updated labels to be consistent with ImageNet class mapping.

import os
import pickle
import numpy as np
from tqdm import tqdm

from .._base import _ClassificationBase


class ImageNet64Dataset(_ClassificationBase):
    
    def __init__(
            self,
            root_dir,
            split = 'train',
            normalize = False,
        ):
        """
        ImageNet64 dataset.

        Parameters
        ----------
        root_dir : str
            Path to the root directory containing the dataset.
            Must contain a "train" and "val" subdirectory,
            each containing the respective data batches.
        split : str, optional
            Which split to load. Must be "train" or "val". Default is "train".
        normalize : bool, optional
            Whether to normalize the images by subtracting the mean and dividing by 255.
            Default is False.

        Attributes
        ----------
        images : np.ndarray
            Array of images in the dataset, with shape (N, 64, 64, 3) and dtype uint8.
        class_indices : list[int]
            List of class indices corresponding to each class in the dataset (0-based).
        class_names : list[str]
            List of class names corresponding to each class in the dataset, sorted by WordNet ID.
        wnid2name : dict[str, str]
            Dictionary mapping WordNet IDs to class names.
        """
        self.root_dir = root_dir
        self.split = split
        self.normalize = normalize

        self.images: np.ndarray
        self.class_indices: list[int]
        self.class_names: list[str]
        self.wnid2name: dict[str, str]

        self._load_data()
        self._load_class_mapping()
        self._update_labels()

        self._initialize()


    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns
        -------
        int
            The number of samples in the dataset.
        """
        return len(self.labels)


    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        """
        Returns the image and label at the specified index.
        
        Parameters
        ----------
        index : int
            Index of the sample to retrieve.
        
        Returns
        -------
        tuple[np.ndarray, int]
            A tuple containing the image (as a numpy array) and its corresponding label (as an integer).
        """
        img = self.images[index]
        label = self.class_name_to_index(self.labels[index])

        return img, label


    def _load_data(self):
        """
        Loads the dataset from the specified root directory and split.
        Expects the data to be organized in batches as follows:
            - root_dir/
                - train/
                    - train_data_batch_1
                    - train_data_batch_2
                    - ...
                - val/
                    - val_data
        """
        if self.split == 'train':

            self.images = np.empty((1281167, 64, 64, 3), dtype=np.uint8)
            self.labels = np.empty((1281167,), dtype=np.int64)
            
            offset = 0
            for i in tqdm(range(1, 11), desc="Loading training data"):
                batch_path = os.path.join(self.root_dir, "train", f"train_data_batch_{i}")
                X_batch, y_batch = self._read_batch(batch_path)
                batch_size = X_batch.shape[0]
                self.images[offset:offset+batch_size] = X_batch
                self.labels[offset:offset+batch_size] = y_batch
                offset += batch_size
            
        elif self.split == 'val':

            self.images, self.labels = self._read_batch(
                os.path.join(self.root_dir, "val", "val_data")
            )

        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train' or 'val'.")
        
    
    def _read_batch(self, batch_path: str) -> tuple[np.ndarray, list[int]]:
        """
        Reads a single batch of data from the specified path.

        Parameters
        ----------
        batch_path : str
            Path to the batch file to read.

        Returns
        -------
        tuple[np.ndarray, list[int]]
            A tuple containing the images (as a numpy array) and
            their corresponding labels (as a list of integers).
        """
        with open(batch_path, "rb") as f:
            batch = pickle.load(f)

        X = batch["data"]
        y = [i - 1 for i in batch["labels"]]  # Convert to 0-based indexing

        X = np.dstack((X[:, :4096], X[:, 4096:8192], X[:, 8192:]))
        X = X.reshape((-1, 64, 64, 3))

        return X, y


    def _load_class_mapping(self):
        """
        Loads the class mapping from the "map_clsloc.txt" file in the root directory.
        Expects the file to be formatted as follows:
            - Each line contains a WordNet ID (wnid), a class index, and a class name, separated by spaces.
            - The class index is 1-based and will be converted to 0-based indexing.
        """
        mapping_path = os.path.join(self.root_dir, "map_clsloc.txt")
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Class mapping file not found at {mapping_path}")

        self.classes, self.class_indices, self.class_names = [], [], []
        with open(mapping_path, "r") as f:
            for line in f:
                wnid, index, class_name = line.strip().split()
                self.classes.append(wnid)
                self.class_indices.append(int(index) - 1)  # Convert to 0-based indexing
                self.class_names.append(class_name)

        # Sort by wnid
        self.classes, self.class_indices, self.class_names = (list(x) for x in zip(*sorted(
            zip(self.classes, self.class_indices, self.class_names),
            key=lambda x: x[0]
        )))


    def _update_labels(self):
        """
        Updates the labels from class indices to class names using the loaded class mapping.
        """
        label_mapping = {k: v for k, v in zip(self.class_indices, self.classes)}
        self.labels = [label_mapping[label] for label in self.labels]


    def _initialize(self):
        super()._initialize()
        self.wnid2name = {wnid: name for wnid, name in zip(self.classes, self.class_names)}
