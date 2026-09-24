"""
Sequential PyTorch models.
"""

# Author: Atif Khurshid
# Created: 2025-06-27
# Modified: 2026-09-24
# Version: 2.0
# Changelog:
#     - 2025-08-01: Added documentation and type hints.
#     - 2026-09-24: Made functionality more similar to nn.Sequential.

from typing import Union
from collections import OrderedDict

import torch
import torch.nn as nn

from .base import PyTorchModel


class PyTorchSequentialModel(PyTorchModel):
    
    def __init__(self, *layers: Union[nn.Module, OrderedDict]):
        """
        Sequential PyTorch model.

        Initializes the model with a list of layers.

        Parameters
        -----------
        layers : Union[nn.Module, OrderedDict]
            Layers to apply in order. Can be individual nn.Module instances
            or an OrderedDict of named layers.
        
        Examples
        ---------
        >>> model = PyTorchSequentialModel([
        ...     nn.Linear(10, 20),
        ...     nn.ReLU(),
        ...     nn.Linear(20, 1)
        ... ])
        """
        super().__init__()

        self._layer_order: list[str] = []
        self._registered: bool = False

        if layers:
            self.register_layers(*layers)
            self._registered = True


    def register_layers(self, *layers: Union[nn.Module, OrderedDict]):
        """
        Register layers to the model.

        Parameters
        -----------
        layers : Union[nn.Module, OrderedDict]
            Layers to apply in order. Can be individual nn.Module instances
            or an OrderedDict of named layers.
        """
        if self._registered:
            raise RuntimeError("Layers have already been registered. Cannot register again.")
        
        if len(layers) == 1 and isinstance(layers[0], OrderedDict):
            names = list(layers[0].keys())
            modules = list(layers[0].values())
        else:
            names = [str(i) for i in range(len(layers))]
            modules = list(layers)

        for name, module in zip(names, modules):
            self.add_module(name, module)

        self._layer_order = names
        self._registered = True


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
        if not self._registered:
            raise RuntimeError("No layers registered. Please call register_layers() before forward().")
        
        for layer_name in self._layer_order:
            x = self._modules[layer_name](x)

        return x
