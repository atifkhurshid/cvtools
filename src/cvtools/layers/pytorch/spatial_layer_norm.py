"""
Spatial Layer Normalization Layer
"""

# Author: Atif Khurshid
# Created: 2025-10-07
# Modified: 2025-11-07
# Version: 1.2
# Changelog:
#    - 2025-10-21: Changed dimensions of affine parameters to match input channels.
#    - 2025-11-07: Improved padding handling to fix edge artifacts.

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialLayerNorm(nn.Module):

    def __init__(
            self,
            in_channels: int,
            summation_kernel_size: int,
            suppression_kernel_size: int,
            affine: bool = True,
            bias: bool = True,
            sigma: float = 0.5,
            trainable_sigma: bool = False,
        ):
        """
        Divisive Normalization layer as described by Ren et al. (2017) in:
        "Normalizing the Normalizers: Comparing and Extending Network Normalization Schemes"

        Adapted from https://www.github.com/renmengye/div-norm
        
        Collects mean and variances on a local window across channels.
        Applies divisive normalization as below:
            x_ = gamma * (x - mean) / sqrt(var + sigma^2) + beta

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        summation_kernel_size : int
            Window size for calculating the mean.
        suppression_kernel_size : int
            Window size for calculating the variance.
        trainable_sigma : bool, optional
            Whether sigma is a learnable parameter, by default False.
        affine : bool, optional
            Whether to include learnable affine parameters gamma and beta, by default False.
        bias : bool, optional
            Whether to include learnable bias parameter beta (only if affine is True), by default False.
        
        Examples
        --------
        >>> divnorm = DivisiveNorm(summation_kernel_size=3, suppression_kernel_size=3)
        >>> x = torch.randn(8, 3, 32, 32)  # Example input tensor
        >>> output = divnorm(x)
        >>> print(output.shape)
        (8, 3, 32, 32)
        """
        super().__init__()

        self.summation_kernel = nn.Parameter(
            self.generate_unit_kernel(summation_kernel_size),
            requires_grad=False
        )
        self.suppression_kernel = nn.Parameter(
            self.generate_unit_kernel(suppression_kernel_size),
            requires_grad=False
        )
        self.summation_padding = [summation_kernel_size // 2] * 4
        self.suppression_padding = [suppression_kernel_size // 2] * 4

        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=False)
        if trainable_sigma:
            self.sigma.requires_grad = True

        self.gamma = None
        self.beta = None
        if affine:
            self.gamma = nn.Parameter(
                torch.ones(in_channels,),
                requires_grad=True
            )
            if bias:
                self.beta = nn.Parameter(
                    torch.zeros(in_channels,),
                    requires_grad=True
                )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Divisive Normalization layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Normalized tensor of the same shape as input.
        """
        x_mean = x.mean(dim=1, keepdim=True)    # Calculate mean across channels first
        x_mean = F.conv2d(                      # Then across spatial dimensions
            F.pad(x_mean, self.summation_padding, mode='replicate'),
            self.summation_kernel
        )
        normed = x - x_mean

        x2 = torch.square(normed)
        x2_mean = x2.mean(dim=1, keepdim=True)
        x2_mean = F.conv2d(
            F.pad(x2_mean, self.suppression_padding, mode='replicate'),
            self.suppression_kernel,
        )
        denom = torch.sqrt(x2_mean + torch.square(self.sigma))
        normed = normed / denom

        if self.gamma is not None:
            normed = normed * self.gamma.view(1, -1, 1, 1)
            if self.beta is not None:
                normed = normed + self.beta.view(1, -1, 1, 1)

        return normed
    

    def generate_unit_kernel(self, size: int) -> torch.Tensor:
        """
        Generate a unit kernel for convolution.
        Output is a size x size kernel with all elements equal to 1/(size*size).

        Parameters
        ----------
        size : int
            Size of the kernel.

        Returns
        -------
        torch.Tensor
            Normalized tensor of the same shape as input.
        """
        kernel = torch.ones((1, 1, size, size)) / (size * size)

        return kernel
