"""
Package: base.dataset
Requirements:
    - numpy
    - PIL
    - pathlib
Use: 
    - from cvtools.base.dataset import ClassificationDataset
Classes:
    - Dataset
"""
from typing import List, Tuple

import numpy as np

from PIL import Image
from pathlib import Path

from ...image.processing import resize


class ClassificationDataset(object):

    def __init__(
            self,
            root_dir: str,
            exts: List[str] = ['.jpg','.png'],
            image_size: Tuple[int, int] = None,
            shuffle: bool = False,
        ) -> None:

        self.root_dir = Path(root_dir)
        self.image_size = image_size

        self.classes = [x.name for x in self.root_dir.iterdir() if not x.is_file()]
        self.class2label = {x : i for i, x in enumerate(self.classes)}
        self.label2class = {v : k for k, v in self.class2label.items()}

        self.labels = []
        self.filenames = []
        for i, c in enumerate(self.classes):
            directory = self.root_dir / c
            filenames = [x.name for x in directory.iterdir() if x.suffix in exts]
            labels = [i] * len(filenames)
            self.filenames.extend(filenames)
            self.labels.extend(labels)

        self.ids = np.arange(len(self.filenames))
        if shuffle:
            np.random.shuffle(self.ids)


    def __len__(self) -> int:
        return len(self.ids)


    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:

        id = self.ids[index]
        X, y = self.__data_generation(id)

        return X, y


    def __data_generation(self, id: int) -> Tuple[np.ndarray, int]:
        
        label = self.labels[id]
        path = self.root_dir / self.label2class[label] / self.filenames[id]
        with Image.open(path) as image:
            if self.image_size:
                image = resize(image, self.image_size)
            image = np.asarray(image)

        return image, label
