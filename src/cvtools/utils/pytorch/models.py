"""
Utility functions for PyTorch models.
"""

# Author: Atif Khurshid
# Created: 2026-11-16
# Modified: None
# Version: 1.0
# Changelog:
#

from typing import Callable

import torch
import torch.nn as nn


def register_forward_hooks(
        model: nn.Module,
        output_layers: list[str],
        output_dict: dict[str, torch.Tensor],
    ):
    """
    Register forward hooks to capture activations from specified modules.

    Parameters
    ----------
    model : nn.Module
        The model from which to capture activations.
    output_layers : list[str]
        A list of layer names from which to capture activations.
    output_dict : dict[str, torch.Tensor]
        A dictionary to store the captured activations.
    """
    def get_hook(name: str) -> Callable:
        def hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            output_dict[name] = output.detach().clone()
        return hook
    
    for name, module in model.named_modules():
        if name in output_layers:
            module.register_forward_hook(get_hook(name))
