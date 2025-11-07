"""
Learnable Divisive Normalization Layer
"""

# Author: Atif Khurshid
# Created: 2025-10-07
# Modified: 2025-11-07
# Version: 1.2
# Changelog:
#    - 2025-10-22: Fixed parameterization and initialization.
#    - 2025-11-07: Fixed weight initialization and edge artifacts.

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableDivisiveNorm(nn.Module):

    def __init__(
            self,
            in_channels: int,
            pool_size: int = 5,
            surround_size: int = 1,
            exponent: bool = False,
        ):
        """
        Divisive Normalization layer described by Burg et al. (2020) in:
        "Learning divisive normalization in primary visual cortex"

        Calculates the average exponentiated values within a spatial neighborhood
        followed by their weighted sum across feature maps to use as divisor.

        Parameters
        ----------
        in_channels : int
            Number of input feature maps.
        pool_size : int
            Window size for spatial averaging.
        surround_size : int, optional
            Size of the spatial surround for summmation of pooled responses.
        exponent : bool, optional
            Whether to learn an exponent parameter, by default False
        
        Examples
        --------
        >>> layer = DivisivePoolNorm(kernel_size=3)
        >>> x = torch.randn(1, 16, 32, 32)
        >>> y = layer(x)
        >>> print(y.shape)
        (1, 16, 32, 32)
        """
        super().__init__()

        self.pool_padding = [pool_size // 2] * 4
        self.surround_padding = [surround_size // 2] * 4

        self.avgpool = nn.AvgPool2d(kernel_size=pool_size, stride=1)
        self.weights = nn.Parameter(
            torch.normal(mean=0, std=1e-3, size=(1, in_channels, surround_size, surround_size)),
            requires_grad=True
        )
        self.semisaturation_constant = nn.Parameter(
            torch.normal(mean=1, std=0.01, size=(in_channels,)),
            requires_grad=True
        )

        if exponent:
            self.exponent = nn.Parameter(
                torch.normal(mean=1, std=0.01, size=(in_channels,)),
                requires_grad=True
            )
        else:
            self.exponent = None
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Divisive Normalization layer.

        Parameters
        ----------
        x : torch.Tensor
            Input feature map of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Normalized feature map of the same shape as input.
        """
        semisaturation_constant = torch.abs(self.semisaturation_constant.view(1, -1, 1, 1))
        if self.exponent is not None:
            exponent = torch.abs(self.exponent.view(1, -1, 1, 1))
            x = torch.pow(x, exponent)
            semisaturation_constant = torch.pow(semisaturation_constant, exponent)

        divisor = self.avgpool(
            F.pad(x, self.pool_padding, mode='replicate')
        )

        divisor = F.conv2d(
            F.pad(divisor, self.surround_padding, mode='replicate'),
            torch.abs(self.weights)
        )
        divisor = divisor + semisaturation_constant

        out = x / (divisor + 1e-6)

        return out
    