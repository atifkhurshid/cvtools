"""
Base class for CNN models.
"""

# Author: Atif Khurshid
# Created: 2025-11-11
# Modified: 2026-09-24
# Version: 1.2
# Changelog:
#     - 2025-11-11: Initial version.
#     - 2025-11-11: Changed init weights to allow submodule initialization only.
#     - 2026-09-24: Moved hooks and weight activation to the base class.

import torch
import torch.nn as nn

from .base import PyTorchModel

class PyTorchCNNModel(PyTorchModel):
    
    def __init__(self):
        """
        Base class for CNN models.
        """
        super().__init__()

        self.features: nn.Module
        self.avgpool: nn.Module
        self.classifier: nn.Module


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.

        The input is sequentially passed through the feature extractor,
        average pooling, and classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)

        return x


    def freeze_backbone(self):
        """
        Freeze the weights of the backbone feature extractor.
        """
        for param in self.features.parameters():
            param.requires_grad = False


    def unfreeze_backbone(self):
        """
        Unfreeze the weights of the backbone feature extractor.
        """
        for param in self.features.parameters():
            param.requires_grad = True
