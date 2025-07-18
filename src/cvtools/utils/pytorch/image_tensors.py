"""
Utility functions for PyTorch image tensors.
"""

# Author: Atif Khurshid
# Created: 2025-06-27
# Modified: None
# Version: 1.0
# Changelog:
#     - None

import torch


def concat_channels_interleaved(x, y):
    # Interleaved concatenation: x, y, x, y
    B, C, H, W = x.shape
    z = torch.stack([x, y], dim=2)
    z = z.view(B, -1, H, W)

    return z


def concat_channels_interleaved_nway(x, y, n):
    # n-way Interleaved concatenation:
    # x1, ... xn, y1, ... yn, xn+1, ... x2n, yn+1, ... yn
    B, C, H, W = x.shape
    C_ = max(1, C // n)
    x = x.view(B, C_, n, H, W)
    y = y.view(B, C_, n, H, W)
    z = torch.concat([x, y], dim=2)
    z = z.view(B, -1, H, W)

    return z
