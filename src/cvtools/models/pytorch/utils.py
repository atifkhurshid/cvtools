"""
Utility functions for PyTorch models.
"""

# Author: Atif Khurshid
# Created: 2025-06-27
# Modified: None
# Version: 1.0
# Changelog:
#     - None

import torch


def concat_interleaved(x, y):
    # Interleaved concatenation: x, y, x, y
    B, C, H, W = x.shape
    z = torch.stack([x, y], dim=2)
    z = z.view(B, -1, H, W)

    return z
