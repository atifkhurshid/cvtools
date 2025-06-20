"""
TensorFlow Wrapper for generic image classification dataloader
"""

# Author: Atif Khurshid
# Created: 2022-12-18
# Modified: 2025-06-18
# Version: 1.0
# Changelog:
#     - 2025-06-18: Updated documentation and type hints.

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from .classification_dataset import ClassificationDataset


class ClassificationDatasetTF(ClassificationDataset, Sequence):
    def __init__(self, *args, **kwargs):
        """
        TensorFlow wrapper class for generic image classification dataset.

        Adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the ClassificationDataset constructor.
        **kwargs : dict
            Keyword arguments passed to the ClassificationDataset constructor.
            - batch_size : int, optional
                Number of samples per batch. Default is 32.
            - shuffle : bool, optional
                Whether to shuffle the dataset at the end of each epoch. Default is False.
        
        Examples
        --------
        >>> from cvtools.datasets.classification.tensorflow import ClassificationDatasetTF
        >>> dataset = ClassificationDatasetTF(root_dir='path/to/dataset', batch_size=32, shuffle=True)
        >>> for images, labels in dataset:
        ...     # Process each batch of images and labels
        ...     pass
        """
        super().__init__(*args, **kwargs)

        self.batch_size = kwargs.get("batch_size", 32)
        self.shuffle = kwargs.get("shuffle", False)

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
