"""
PyTorch wrapper for Princeton SUN dataloader.

Author: Atif Khurshid
Created: 2025-05-23
Modified: None
Version: 1.0

Changelog:
    - None
"""

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ToImage
from ...base import SUNDataset


class SUNDatasetPT(SUNDataset, Dataset):

    def __init__(
            self,
            root_dir,
            class_hierarchy = "basic",
            image_size = None,
            train = True,
            transform = None,
            target_transform = None,
        ):
        super().__init__(
            root_dir = root_dir,
            class_hierarchy = class_hierarchy,
            image_size = image_size,
            train = train,
        )

        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        image = ToImage()(image)
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
