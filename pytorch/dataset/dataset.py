"""
Package: pytorch.dataset
Requirements:
    - torch
    - numpy
    - PIL
    - pathlib
Use: 
    - from cvtools.pytorch.dataset import Dataset
Classes:
    - Dataset
"""
import torch
import numpy as np

from PIL import Image
from pathlib import Path


class Dataset(torch.utils.data.Dataset):

    def __init__(self, root, exts=['.jpg','.png'], image_size=None, transform=None):

        self.root = Path(root)
        self.images_dir = self.root / 'images'
        self.image_size = image_size
        self.transform = transform

        self.filenames = [x.name for x in self.images_dir.iterdir() if x.suffix in exts]
        self.ids = np.arange(len(self.filenames))


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):

        id = self.ids[index]
        X = self._data_generation(id)
        if self.transform:
            X = self.transform(X)

        return X


    def _data_generation(self, id):

        path = self.images_dir / self.filenames[id]
        with Image.open(path) as image:
            if self.image_size:
                image = image.resize(self.image_size)
            image = np.asarray(image)
        image = torch.from_numpy(image)
        
        return image
