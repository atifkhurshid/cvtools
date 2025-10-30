"""
TensorFlow Wrapper for generic image classification dataloader
"""

# Author: Atif Khurshid
# Created: 2022-12-18
# Modified: 2025-10-30
# Version: 1.1
# Changelog:
#     - 2025-06-18: Updated documentation and type hints.
#     - 2025-10-30: Updated arguments to match base class.

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from .classification_dataset import ClassificationDataset


class ClassificationDatasetTF(ClassificationDataset, Sequence):
    def __init__(
            self,
            root_dir: str,
            exts: list[str] = ['.jpg', '.png'],
            image_mode: str | int = 'RGB',
            image_size: tuple[int, int] | None = None,
            preserve_aspect_ratio: bool = True,
            batch_size: int = 32,
            shuffle: bool = False,
        ):
        """
        TensorFlow wrapper class for generic image classification dataset.

        Adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

        Parameters
        ----------
        root_dir : str
            Path to the root directory containing class subdirectories.
        exts : list[str], optional
            List of file extensions to consider as valid images. Default is ['.jpg', '.png'].
        image_mode : str | int, optional
            Mode to read images. Can be 'RGB', 'GRAY', or a cv2.IMREAD_... flag. Default is 'RGB'.
        image_size : tuple[int, int] | None, optional
            Size to which images will be resized. If None, images will not be resized. Default is None.
        preserve_aspect_ratio : bool, optional
            If True, images will be resized while preserving their aspect ratio. Default is True.
        batch_size : int, optional
            Number of samples per batch. Default is 32.
        shuffle : bool, optional
            Whether to shuffle the dataset at the end of each epoch. Default is False.

        Examples
        --------
        >>> from cvtools.datasets.classification.tensorflow import ClassificationDatasetTF
        >>> dataset = ClassificationDatasetTF(root_dir='path/to/dataset', batch_size=32, shuffle=True)
        >>> for images, labels in dataset:
        ...     # Process each batch of images and labels
        ...     pass
        """
        super().__init__(root_dir, exts, image_mode, image_size, preserve_aspect_ratio)

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()


    def __len__(self) -> int:
        """
        Returns the number of batches in the dataset.
        The length is calculated as the total number of samples divided by the batch size,
        rounded up to the nearest integer.

        Returns
        -------
        int
            The number of batches in the dataset.
        """
        return int(tf.math.ceil(len(self.ids) / self.batch_size))


    def __getitem__(self, index: int) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Generate one batch of data.

        Parameters
        ----------
        index : int
            Index of the batch to retrieve.

        Returns
        -------
        tuple[tf.Tensor, tf.Tensor]
            A tuple containing:
            - images: A tensor of shape (batch_size, height, width, channels) containing the images.
            - labels: A tensor of shape (batch_size,) containing the corresponding labels.
        """
        ids = self.ids[index*self.batch_size:(index+1)*self.batch_size]

        images = []
        labels = []
        for id in ids:
            image, label = super().__getitem__(id)
            images.append(image)
            labels.append(label)
        images = tf.convert_to_tensor(np.array(images), dtype=tf.int32)
        labels = tf.convert_to_tensor(np.array(labels), dtype=tf.int32)
        
        return images, labels


    def on_epoch_end(self):
        """
        Shuffle the dataset at the end of each epoch if shuffle is set to True.
        """
        if self.shuffle == True:
            self.ids = tf.random.shuffle(self.ids).numpy()
