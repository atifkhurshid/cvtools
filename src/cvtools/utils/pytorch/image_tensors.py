"""
Utility functions for PyTorch image tensors.
"""

# Author: Atif Khurshid
# Created: 2025-06-27
# Modified: 2025-08-01
# Version: 1.1
# Changelog:
#     - 2025-08-01: Added documentation and type hints.

import torch


def concat_channels_interleaved(
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
    """
    Concatenate two tensors along the channel dimension in an interleaved manner.

    Interleaved concatenation: x, y, x, y

    Parameters
    ----------
    x : torch.Tensor
        The first input tensor.
    y : torch.Tensor
        The second input tensor.

    Returns
    -------
    torch.Tensor
        The interleaved concatenation of the input tensors.
    
    Examples
    ---------
    >>> x = torch.randn(2, 3, 4, 4)
    >>> y = torch.randn(2, 3, 4, 4)
    >>> z = concat_channels_interleaved(x, y)
    >>> print(z.shape)
    torch.Size([2, 6, 4, 4])
    """
    B, C, H, W = x.shape
    z = torch.stack([x, y], dim=2)
    z = z.view(B, -1, H, W)

    return z


def concat_channels_interleaved_nway(
        x: torch.Tensor,
        y: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
    """
    Concatenate two tensors along the channel dimension in an n-way interleaved manner.

    n-way Interleaved concatenation: x1, ... xn, y1, ... yn, xn+1, ... x2n, yn+1, ... yn

    Parameters
    ----------
    x : torch.Tensor
        The first input tensor.
    y : torch.Tensor
        The second input tensor.
    n : int
        The number of interleaved segments.

    Returns
    -------
    torch.Tensor
        The n-way interleaved concatenation of the input tensors.
    
    Examples
    ---------
    >>> x = torch.randn(2, 6, 4, 4)
    >>> y = torch.randn(2, 6, 4, 4)
    >>> z = concat_channels_interleaved_nway(x, y, n=2)
    >>> print(z.shape)
    torch.Size([2, 12, 4, 4])
    """
    B, C, H, W = x.shape
    C_ = max(1, C // n)
    x = x.view(B, C_, n, H, W)
    y = y.view(B, C_, n, H, W)
    z = torch.concat([x, y], dim=2)
    z = z.view(B, -1, H, W)

    return z
