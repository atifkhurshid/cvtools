"""
Package: pytorch.dataset
Requirements:
    - torch
    - numpy
    - PIL
    - pathlib
Use: 
    - from cvtools.pytorch.dataset import ClassificationDatasetPT
Classes:
    - ClassificationDatasetPT
"""
import torch
from torch.utils.data import Dataset

from ...base import ClassificationDataset


class ClassificationDatasetPT(ClassificationDataset, Dataset):

    def __init__(
            self,
            root_dir: str,
            exts: list[str] = ['.jpg','.png'],
            image_size: tuple[int, int] = (224, 224),
            transform = None,
        ) -> None:

        super().__init__(root_dir, exts, image_size)

        self.transform = transform


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:

        id = self.ids[index]

        X, y = self._data_generation(id)

        if self.transform:
            X = self.transform(X)

        return X, y


    def _data_generation(self, id: int) -> tuple[torch.Tensor, torch.Tensor]:

        image, label = super()._data_generation(id)

        image = torch.from_numpy(image)
        label = torch.tensor(label, dtype=torch.int32)
        
        return image, label
