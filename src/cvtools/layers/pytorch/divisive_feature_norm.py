"""
Divisive Feature Normalization Layer
"""

# Author: Atif Khurshid
# Created: 2025-10-07
# Modified: 2025-10-23
# Version: 1.1
# Changelog:
#    - 2025-10-23: Updated parameters and docstrings.

import torch
import torch.nn as nn
import torch.nn.functional as F


class DivisiveFeatureNorm(nn.Module):

    def __init__(self, initialization: str = "divisive"):
        """
        Divisive Normalization layer as described by Miller et al. (2022) in:
        "Divisive Feature Normalization Improves Image Recognition Performance in AlexNet"

        Adapted from: https://github.com/MikiMiller95/DivisiveNormalization

        Parameters
        ----------
        initialization : str, optional
            Type of initialization for the parameters.
            Options are "divisive" or "combined", by default "divisive".
        
        Examples
        --------
        >>> import torch
        >>> layer = DivisiveFeatureNorm(initialization="divisive")
        >>> input_tensor = torch.randn(1, 3, 32, 32)
        >>> output_tensor = layer(input_tensor)
        >>> print(output_tensor.shape)
        torch.Size([1, 3, 32, 32])
        """
        super().__init__()

        alpha = 0.1
        beta = 1.0
        if initialization == "divisive":
            lambda_ = 1.0
            k = 0.5
        elif initialization == "combined":
            lambda_ = 10.0
            k = 10.0
        else:
            raise ValueError("Invalid initialization type. Choose 'divisive' or 'combined'.")

        self.lambda_ = nn.Parameter(torch.Tensor([lambda_]))
        self.alpha = nn.Parameter(torch.Tensor([alpha])) 
        self.beta = nn.Parameter(torch.Tensor([beta])) 
        self.k = nn.Parameter(torch.Tensor([k]))
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Divisive Normalization layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Output tensor after applying divisive normalization.
        """
        neighbors = int(torch.ceil(2*4*self.lambda_).item())
        if neighbors % 2 == 0:
            neighbors = neighbors + 1

        dim = x.dim()
        if dim < 3:
            raise ValueError('Expected 3D or higher dimensionality \
                            input (got {} dimensions)'.format(dim))
        div = x.mul(x).unsqueeze(1)
        # hacky trick to try and keep everything on cuda
        sizes = x.size()
        weits = x.clone().detach() 
        weits = weits.new_zeros(([1]+[1]+[int(neighbors)]+[1]+[1]))

        if dim == 3:
            div = F.pad(div, (0, 0, neighbors // 2,  neighbors - 1 // 2))
            div = torch._C._nn.avg_pool2d((div,  neighbors, 1), stride=1).squeeze(1)
        else:
            dev = x.get_device()
            # indexx is a 1D tensor that is a symmetric exponential distribution of some "radius" neighbors
            idxs = torch.abs(torch.arange(neighbors)-neighbors//2)
            weits[0,0,:,0,0] = idxs
            weits = torch.exp(-weits/self.lambda_)
            # creating single dimension at 1;corresponds to number of input channels;only 1 input channel
            # 3D convolution; weits has dims: Cx1xCx1x1 ; this means we have C filters for the C channels
            # The div is the input**2; it has dimensions B x 1 x C x W x H
            div = F.conv3d(div, weits, padding=((neighbors // 2), 0, 0))
            div = div / self.lambda_

        div = div.mul(self.alpha).add(1).mul(self.k).pow(self.beta)
        div = div.squeeze()
        
        return x.mul(x) / div
    