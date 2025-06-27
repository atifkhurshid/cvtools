"""
Sequential PyTorch models.
"""

# Author: Atif Khurshid
# Created: 2025-06-27
# Modified: None
# Version: 1.0
# Changelog:
#     - None

import torch
import torch.nn as nn

from .model import PyTorchModel


class PyTorchSequentialModel(PyTorchModel):
    
    def __init__(self, layers: list = []):
        super().__init__()

        for layer in layers:
            self.layers.append(layer)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
