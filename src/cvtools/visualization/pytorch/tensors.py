"""
Visualizations for Pytorch Tensors.
"""

# Author: Atif Khurshid
# Created: 2025-08-29
# Modified: None
# Version: 1.0
# Changelog:
#     - None

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils


def visualize_tensor(
        tensor: torch.Tensor,
        channel: int = 0,
        all_channels: bool = False,
        normalize: bool = True,
        ncols: int = 8,
        padding: int = 1,
    ):
    """
    Visualize a Pytorch tensor.

    Adapted from https://stackoverflow.com/a/55604568

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of shape (N, C, H, W).
    channel : int, default=0
        Channel index to visualize if all_channels is False.
    all_channels : bool, default=False
        Whether to visualize all channels or a specific channel.
    normalize : bool, default=True
        Whether to normalize the tensor values to [0, 1].
    ncols : int, default=8
        Number of columns in the grid.
    padding : int, default=1
        Padding between images in the grid.

    Examples
    --------
    >>> tensor = torch.randn(16, 3, 32, 32)
    >>> visualize_tensor(tensor, channel=0)
    ...
    >>> visualize_tensor(tensor, all_channels=True)
    ...
    """
    n, c, h, w = tensor.size()

    if all_channels:
        tensor = tensor.view(n * c, -1, h, w)
    else:
        tensor = tensor[:, channel, :, :].unsqueeze(1)

    if normalize:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-5)

    nrows = np.min((tensor.shape[0] // ncols + 1, 64))
    grid = utils.make_grid(tensor, nrow=ncols, normalize=False, padding=padding)
    grid = grid.numpy().transpose((1, 2, 0))    # Change to (H, W, C) format of image

    plt.figure(figsize=(ncols, nrows))
    plt.imshow(grid)
    plt.axis('off')
    plt.ioff()
    plt.show()
