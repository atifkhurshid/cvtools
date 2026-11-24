"""
Base class for custom layers in PyTorch.
"""

# Author: Atif Khurshid
# Created: 2026-11-24
# Modified: None
# Version: 1.0
# Changelog:
#

import inspect

import torch
import torch.nn as nn


class nnModule(nn.Module):
    """
    Base class for custom layers in PyTorch.

    This class serves as a base for creating custom layers in PyTorch. It extends
    the nn.Module class and provides a default extra_repr method for printing layer information.
    """
    def __init__(self, *args, **kwargs):
        super(nnModule, self).__init__(*args, **kwargs)


    def extra_repr(self) -> str:
        """
        Returns a string representation of the layer's parameters.
        """
        params = inspect.signature(type(self).__init__).parameters
        names = [n for n in params if n != "self" and hasattr(self, n)]
        return ", ".join(f"{n}={getattr(self, n)!r}" for n in names)
    