"""
Base class for CNN models.
"""

# Author: Atif Khurshid
# Created: 2025-11-11
# Modified: 2025-11-11
# Version: 1.1
# Changelog:
#     - 2025-11-11: Initial version.
#     - 2025-11-11: Changed init weights to allow submodule initialization only.

import math
from typing import Callable

import torch
import torch.nn as nn

from .base import PyTorchModel

class PyTorchCNNModel(PyTorchModel):
    
    def __init__(self):
        """
        Base class for CNN models.
        """
        super().__init__()

        self.activations = {}
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


    def init_weights(
            self,
            model: nn.Module,
            nonlinearity: str = 'relu',
            a: float = 0,
        ):
        """
        Initialize the weights of the model using Kaiming uniform initialization.

        Parameters
        ----------
        model : nn.Module
            The model or submodule whose weights are to be initialized.
        nonlinearity : str, optional
            The non-linear function used after convolutional layers, by default 'relu'.
        a : float, optional
            The negative slope of the rectifier used after this layer (only
            used with 'leaky_relu'), by default 0.
        """
        for module in model.modules():
            
            bias = False
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=a, nonlinearity=nonlinearity)
                if module.bias is not None:
                    bias = True

            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    bias = True

            if bias:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound)


    def register_hooks(self, modules_to_hook: list[str]):
        """
        Register forward hooks to capture activations from specified modules.

        Parameters
        ----------
        modules_to_hook : list[str]
            List of module names to register hooks on.
        """
        def get_hook(name: str) -> Callable:
            def hook(
                    module: nn.Module,
                    input: torch.Tensor,
                    output: torch.Tensor
                ):
                self.activations[name] = output.detach().clone().cpu()
                
            return hook
        
        for name, module in self.named_modules():
            if name in modules_to_hook:
                module.register_forward_hook(get_hook(name))


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


    def replace_inplace_relus(self, module: nn.Module):
        """
        Recursively replace all inplace ReLU activations
        with non-inplace versions.

        Parameters
        ----------
        module : nn.Module
            The module in which to replace inplace ReLUs.
        """
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU):
                setattr(module, name, nn.ReLU(inplace=False))
            else:
                self.replace_inplace_relus(child)
    
