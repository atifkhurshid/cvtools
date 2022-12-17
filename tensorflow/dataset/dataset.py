"""
Package: tensorflow.dataset
Requirements:
    - tensorflow
    - numpy
    - PIL
    - pathlib
Use: 
    - from cvtools.tensorflow.dataset import Dataset
Classes:
    - Dataset

# Source: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
"""
import numpy as np
import tensorflow as tf

from PIL import Image
from pathlib import Path


class Dataset(tf.keras.utils.Sequence):

    def __init__(self, root, exts=['.jpg','.png'], image_size=None, batch_size=32, shuffle=False):

        self.root = Path(root)
        self.images_dir = self.root / 'images'
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.filenames = [x.name for x in self.images_dir.iterdir() if x.suffix in exts]
        self.ids = tf.range(len(self.filenames))

        self.on_epoch_end()


    def __len__(self):
        return int(tf.math.ceil(len(self.ids) / self.batch_size))


    def __getitem__(self, index):

        ids = self.ids[index*self.batch_size:(index+1)*self.batch_size]

        X = self.__data_generation(ids)

        return X


    def __data_generation(self, ids):

        images = []
        for id in ids:
            path = self.images_dir / self.filenames[id]
            with Image.open(path) as image:
                if self.image_size:
                    image = image.resize(self.image_size)
                image = np.asarray(image)
            images.append(image)
        images = np.array(images)
        images = tf.convert_to_tensor(images, dtype=tf.int32)
        
        return images


    def on_epoch_end(self):
        if self.shuffle == True:
            tf.random.shuffle(self.ids)
