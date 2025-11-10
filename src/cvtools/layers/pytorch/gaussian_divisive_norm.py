"""
Gaussian Divisive Normalization Layer
"""

# Author: Atif Khurshid
# Created: 2025-06-25
# Modified: 2025-11-10
# Version: 1.5
# Changelog:
#     - 2025-08-01: Added documentation and type hints.
#     - 2025-10-06: Made beta parameter trainable.
#     - 2025-10-24: Updated weight initialization and fixed a few bugs.
#     - 2025-11-07: Improved padding handling to fix edge artifacts.
#     - 2025-11-10: Fixed kernel generation according to the original paper.

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianDivisiveNorm(nn.Module):

    def __init__(
            self,
            in_channels: int,
            kernel_size: int,
            sigma: float = 1,
            stride: int = 1,
            padding: Union[int, str] = "auto",
            trainable: bool = False,
        ):
        """
        Gaussian Divisive Normalization as described by Cirincione et al. (2022) in:
        "Implementing Divisive Normalization in CNNs Improves Robustness to Common Image Corruptions"

        Applies Gaussian filter to each input channel and normalizes the input by the sum of the filtered channels.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        kernel_size : int
            Size of the square kernel.
        sigma : float, optional
            Standard deviation of the Gaussian kernel. Default is 1.
        beta : float, optional
            Small constant to avoid division by zero. Default is 1e-5.
        stride : int, optional
            Stride for the convolution operation. Default is 1.
        padding : int or str, optional
            Padding for the convolution operation. If "auto", padding is set to kernel_size // 2.
            Default is "auto".
        trainable : bool, optional
            If True, the parameters of the Gaussian kernel will be trainable. Default is False.

        Attributes
        ----------
        padding : tuple
            Padding for the convolution operation, calculated based on kernel size.
        weight : torch.nn.Parameter
            Filter weights for the Gaussian kernel, initialized from the provided parameters.
        out_channels : int
            Number of output channels after applying the Gaussian divisive normalization, which is equal to the number
            of input channels.
        
        Examples
        --------
        >>> gdn = GaussianDivisiveNorm(in_channels=3, kernel_size=15, sigma=3, beta=1e-5, stride=1)
        >>> input_tensor = torch.randn(1, 3, 32, 32)
        >>> output_tensor = gdn(input_tensor)
        >>> output_tensor.shape
        torch.Size([1, 3, 32, 32])  # Output shape
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = (stride, stride)
        self.trainable = trainable

        if padding == "auto":
            self.padding = [kernel_size // 2] * 4
        else:
            self.padding = [padding] * 4

        self.sigma_x = self._create_parameter(sigma, torch.float, trainable)
        self.sigma_y = self._create_parameter(sigma, torch.float, trainable)
        self.theta = self._create_parameter(0, torch.float, trainable)
        self.A = self._create_parameter(1, torch.float, trainable)
        self.u = self._create_parameter(0, torch.float, trainable)
        self.v = self._create_parameter(0, torch.float, trainable)
        self.beta = nn.Parameter(torch.zeros(in_channels,), requires_grad=trainable)

        if trainable:
            self._init_weights(self.sigma_x, sigma)
            self._init_weights(self.sigma_y, sigma)
            self._init_weights(self.theta, 0)
            self._init_weights(self.A, 1)
            self._init_weights(self.u, 0)
            self._init_weights(self.v, 0)
            self._init_weights(self.beta, 0)

        self.weight = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Gaussian Divisive Normalization.

        Generates a Gaussian kernel for each input channel and applies it to the input tensor channelwise.
        The output tensor is normalized by the sum of the filtered channels plus a small constant `beta`.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor after applying the Gaussian divisive normalization.
            The output tensor has the same shape as the input tensor.
        """
        if self.weight is None or self.trainable:
            self.weight = self._generate_gaussian_kernel()

        div = F.conv2d(
            F.pad(x, self.padding, mode='reflect'),
            self.weight, stride=self.stride
        )
        div = div + torch.abs(self.beta).view(1, -1, 1, 1)

        x = x / (div + 1e-6)

        return x


    def _create_parameter(
            self,
            value: float,
            dtype: type, 
            trainable: bool,
        ) -> nn.Parameter:
        """
        Creates a trainable or non-trainable parameter based on the provided value and dtype.

        Parameters
        ----------
        value : float
            The initial value for the parameter.
        dtype : type
            The data type of the parameter.
        trainable : bool
            If True, the parameter will be trainable.
        """
        data = torch.ones((self.in_channels, self.in_channels), dtype=dtype) * value

        return nn.Parameter(data, trainable)
    

    def _init_weights(self, parameter: nn.Parameter, mean: float):
        """
        Initializes the weights of a parameter with a normal distribution.

        Parameters
        ----------
        parameter : nn.Parameter
            The parameter to initialize.
        mean : float
            The mean of the normal distribution.
        """
        nn.init.normal_(parameter, mean=mean, std=0.05)


    def _generate_gaussian_kernel(self):
        """
        Generates the weights of the Gaussian kernels based on current parameters.
        """
        # Adapted from: github.com/dicarlolab/vonenet
        view_shape = (self.in_channels, self.in_channels, 1, 1)
        sigma_x = torch.abs(self.sigma_x).view(view_shape) + 1e-5
        sigma_y = torch.abs(self.sigma_y).view(view_shape) + 1e-5
        theta = self.theta.view(view_shape)
        A = torch.abs(self.A).view(view_shape)
        u = self.u.view(view_shape)
        v = self.v.view(view_shape)

        device = sigma_x.device

        n = self.kernel_size // 2
        y, x = torch.meshgrid(
            torch.arange(-n, n + 1, device=device),
            torch.arange(-n, n + 1, device=device),
            indexing='ij'
        )
        x = x.repeat(view_shape)
        y = y.repeat(view_shape)

        ct = torch.cos(theta)
        st = torch.sin(theta)
        rotx = x * ct + y * st
        roty = -x * st + y * ct

        gaussian = (A / (2 * torch.pi * sigma_x * sigma_y)) * torch.exp(
            -0.5 * ((rotx + u)**2 / sigma_x**2 + (roty + v)**2 / sigma_y**2))

        return gaussian
