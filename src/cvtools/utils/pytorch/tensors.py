"""
Utility functions for PyTorch tensors.
"""

# Author: Atif Khurshid
# Created: 2026-11-23
# Modified: None
# Version: 1.0
# Changelog:
#

import torch


def minmax_normalization(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor using min-max normalization.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor to be normalized.

    Returns
    -------
    torch.Tensor
        The normalized tensor with values scaled to the range [0, 1].
    """
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)

    return normalized_tensor
