"""
64x64 ImageNet dataset.
"""

# Author: Atif Khurshid
# Created: 2026-03-26
# Modified: 2026-03-27
# Version: 1.0
# Changelog:
#     - 2026-03-26: Created 64x64 ImageNet dataset class.
#     - 2026-03-27: Refactored code to match updated base class.

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
        """
        self.root_dir = root_dir
        self.split = split
        self.normalize = normalize

        self.images: np.ndarray
        self.labels: np.ndarray

        self._load_data()

        self.classes = np.unique(self.labels)

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
        return self.images[index], self.labels[index]


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
        
    
    def _read_batch(self, batch_path: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Reads a single batch of data from the specified path.

        Parameters
        ----------
        batch_path : str
            Path to the batch file to read.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            A tuple containing the images (as a numpy array) and
            their corresponding labels (as a numpy array of integers).
        """
        with open(batch_path, "rb") as f:
            batch = pickle.load(f)

        X = batch["data"]
        y = np.array(batch["labels"]) - 1  # Convert to 0-based indexing

        X = np.dstack((X[:, :4096], X[:, 4096:8192], X[:, 8192:]))
        X = X.reshape((-1, 64, 64, 3))

        return X, y
    