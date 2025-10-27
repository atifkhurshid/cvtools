"""
Sequential PyTorch models.
"""

# Author: Atif Khurshid
# Created: 2025-06-27
# Modified: 2025-08-01
# Version: 1.1
# Changelog:
#     - 2025-08-01: Added documentation and type hints.

import torch
import torch.nn as nn

from .base import PyTorchModel


class PyTorchSequentialModel(PyTorchModel):
    
    def __init__(self, layers: list[nn.Module] = []):
        """
        Sequential PyTorch model.

        Initializes the model with a list of layers.

        Parameters
        -----------
        layers : list[nn.Module]
            List of layers to add to the model.
        
        Examples
        ---------
        >>> model = PyTorchSequentialModel([
        ...     nn.Linear(10, 20),
        ...     nn.ReLU(),
        ...     nn.Linear(20, 1)
        ... ])
        >>> print(model)
        >>> X = torch.randn(5, 10)
        >>> output = model(X)
        >>> print(output.shape)
        torch.Size([5, 1])
        """
        super().__init__()

        self.layers = nn.ModuleList()
        for layer in layers:
            self.layers.append(layer)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Sequentially applies each layer to the input tensor.

        Parameters
        -----------
        x : torch.Tensor
            Input tensor.

        Returns
        --------
        torch.Tensor
            Output tensor.
        """
        for layer in self.layers:
            x = layer(x)

        return x
