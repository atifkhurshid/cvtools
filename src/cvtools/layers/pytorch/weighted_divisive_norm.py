"""
Weighted Divisive Normalization Layer
"""

# Author: Atif Khurshid
# Created: 2025-10-23
# Modified: 2025-11-07
# Version: 1.1
# Changelog:
#    - 2025-11-07: Improved padding handling to fix edge artifacts.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedDivisiveNorm(nn.Module):

    def __init__(
            self,
            in_channels: int,
            group_size: int = 8,
            surround: bool = False,
            surround_size: int = 3,
            affine: bool = True,
            bias: bool = True,
        ):
        """
        Divisive Normalization layer as described by Pan et al. (2021) in:
        "Brain-inspired Weighted Normalization for CNN Image Classification"

        Uses separable convolutions to compute center and surround normalization pools.
        Applies divisive normalization as below:
            x_ = gamma * x / sqrt(w.x^2 + v.x^2 + surround_sum + epsilon) + beta

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        group_size : int, optional
            Size of the center group for normalization, by default 8.
        surround : bool, optional
            Whether to include surround normalization, by default False.
        surround_size : int, optional
            Size of the surround kernel, by default 3.
        affine : bool, optional
            Whether to include learnable affine parameters gamma and beta, by default True.
        bias : bool, optional
            Whether to include learnable bias parameter beta (only if affine is True), by default True
        
        Examples
        --------
        >>> divnorm = WeightedDivisiveNorm(in_channels=16, group_size=8, surround=True)
        >>> x = torch.randn(8, 16, 32, 32)  # Example input tensor
        >>> output = divnorm(x)
        >>> print(output.shape)
        (8, 16, 32, 32)
        """
        super().__init__()

        self.in_channels = in_channels
        self.groups = in_channels // group_size

        self.center_weight = nn.Parameter(
            torch.empty((in_channels, group_size, 1, 1)),
            requires_grad=True
        )
        nn.init.kaiming_uniform_(self.center_weight, a=math.sqrt(5))

        self.surround_weight = None
        if surround:
            self.surround_weight = nn.Parameter(
                torch.empty((in_channels, 1, surround_size, surround_size)),
                requires_grad=True
            )
            nn.init.kaiming_uniform_(self.surround_weight, a=math.sqrt(5))
            self.surround_padding = [surround_size // 2] * 4

        self.gamma = None
        self.beta = None
        if affine:
            self.gamma = nn.Parameter(torch.ones(in_channels,), requires_grad=True)
            if bias:
                self.beta = nn.Parameter(torch.zeros(in_channels,), requires_grad=True)

        self.epsilon = nn.Parameter(torch.empty((in_channels,)), requires_grad=True)
        nn.init.uniform_(self.epsilon, a=1e-6, b=1e-3)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Weighted Divisive Normalization layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Normalized tensor of the same shape as input.
        """
        x2 = torch.square(x)
        center_weight = torch.abs(self.center_weight)
        div = F.conv2d(x2, center_weight, groups=self.groups)

        if self.surround_weight is not None:
            surround_weight = torch.abs(self.surround_weight)
            x2 = F.pad(x2, self.surround_padding, mode='reflect')
            div = div + F.conv2d(x2, surround_weight, groups=self.in_channels)

        div = div + torch.abs(self.epsilon.view(1, -1, 1, 1))
        div = torch.sqrt(div)

        normed = x / (div + 1e-6)

        if self.gamma is not None:
            normed = normed * self.gamma.view(1, -1, 1, 1)

        if self.beta is not None:
            normed = normed + self.beta.view(1, -1, 1, 1)

        return normed
