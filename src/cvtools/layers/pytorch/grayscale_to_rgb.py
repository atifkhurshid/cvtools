"""
Grayscale to RGB Layer
"""

# Author: Atif Khurshid
# Created: 2026-06-17
# Modified: None
# Version: 1.0
# Changelog:
#     - 2026-06-17: Initial version.

import torch
import torch.nn as nn


class GrayscaleToRGB(nn.Module):

    def __init__(self):
        """
        Layer to convert grayscale images to RGB by replicating the single channel.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert grayscale images to RGB by replicating the single channel.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, H, W).
        
        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, 3, H, W).
        """
        return x.repeat(1, 3, 1, 1)
    