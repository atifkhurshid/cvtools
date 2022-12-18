"""
Package: tensorflow.dataset
Requirements:
    - tensorflow
    - numpy
    - PIL
    - pathlib
Use: 
    - from cvtools.tensorflow.dataset import ClassificationDatasetTF
Classes:
    - ClassificationDatasetTF

# Source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from ...base.dataset import ClassificationDataset


class ClassificationDatasetTF(ClassificationDataset, tf.keras.utils.Sequence):

    def __init__(
            self,
            root_dir: str,
            exts: List[str] = ['.jpg','.png'],
            image_size: Tuple[int, int] = (224, 224),
            batch_size: int = 32,
            shuffle: bool = False
        ) -> None:

        super().__init__(root_dir, exts, image_size)

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()


    def __len__(self) -> int:
        return int(tf.math.ceil(len(self.ids) / self.batch_size))


    def __getitem__(self, index: int) -> Tuple[tf.Tensor, tf.Tensor]:

        ids = self.ids[index*self.batch_size:(index+1)*self.batch_size]

        X, y = self._data_generation(ids)

        return X, y


    def _data_generation(self, ids: np.ndarray) -> Tuple[tf.Tensor, tf.Tensor]:

        images = []
        labels = []
        for id in ids:
            image, label = super()._data_generation(id)
            images.append(image)
            labels.append(label)
        images = tf.convert_to_tensor(np.array(images), dtype=tf.int32)
        labels = tf.convert_to_tensor(np.array(labels), dtype=tf.int32)
        
        return images, labels


    def on_epoch_end(self):
        if self.shuffle == True:
            self.ids = tf.random.shuffle(self.ids).numpy()
