"""
Base class for PyTorch models.
"""

# Author: Atif Khurshid
# Created: 2025-06-22
# Modified: None
# Version: 1.0
# Changelog:
#     - None

import torch
import torch.nn as nn


class PyTorchModel(nn.Module):
    
    def __init__(self):
        """
        Base class for PyTorch models.
        """
        super().__init__()

        self.layers = nn.ModuleList()


    def train_step(self, X, y):
        raise NotImplementedError("This model is not trainable.")
    

    def eval_step(self, X, y):
        raise NotImplementedError("This model is not evaluatable.")
