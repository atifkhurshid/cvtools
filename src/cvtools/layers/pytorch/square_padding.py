"""
Squared Padding Layer
"""

# Author: Atif Khurshid
# Created: 2026-03-03
# Modified: None
# Version: 1.0
# Changelog:
#    - 2026-03-03: Initial implementation of SquarePad2D layer.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SquarePad2D(nn.Module):

    def __init__(self, **kwargs):
        """
        PyTorch Layer that pads a tensor to make it square
        by adding equal padding on both sides of the shorter dimension.

        Parameters
        ----------
        **kwargs: Additional keyword arguments to pass to F.pad (e.g., mode, value)
        """
        super().__init__()
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pad the input tensor to make it square.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, channels, height, width)

        Returns
        -------
        torch.Tensor
            Padded tensor of shape (batch_size, channels, max(height, width), max(height, width))
        """
        h, w = x.shape[-2:]

        diff = abs(h - w)

        top = bottom = left = right = 0
        if h < w:
            top = diff // 2
            bottom = diff - top
        else:
            left = diff // 2
            right = diff - left
        
        return F.pad(x, (left, right, top, bottom), **self.kwargs)
    