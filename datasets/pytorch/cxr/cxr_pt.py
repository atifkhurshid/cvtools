"""
PyTorch wrapper for NIH Chest X-Ray dataloader.

Author: Atif Khurshid
Created: 2025-05-22
Modified: None
Version: 1.0

Changelog:
    - None
"""

import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import ToImage
from ...base import CXRDataset


class CXRDatasetPT(CXRDataset, Dataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transform = kwargs.get('transform', None)
        self.target_transform = kwargs.get('target_transform', None)

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        image = ToImage()(image).to(torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

