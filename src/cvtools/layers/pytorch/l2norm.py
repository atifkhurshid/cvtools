"""
PyTorch Layers.
"""

# Author: Atif Khurshid
# Created: 2025-09-02
# Modified: None
# # Version: 1.0
# Changelog:
#     - None

import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Norm(nn.Module):

    def __init__(self, dim=1):
        """
        L2 Normalization layer.

        Parameters
        ----------
        dim : int, optional
            The dimension along which to normalize, by default 1
        """
        super().__init__()

        self.dim = dim


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for L2 normalization.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to normalize.

        Returns
        -------
        torch.Tensor
            Normalized tensor.

        Examples
        --------
        >>> layer = L2Norm(dim=1)
        >>> x = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
        >>> layer(x)
        tensor([[0.6000, 0.8000],
                [0.4472, 0.8944]])
        """
        x = F.normalize(x, p=2, dim=self.dim)

        return x
    