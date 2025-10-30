"""
Pytorch image processing module.
"""

# Author: Atif Khurshid
# Created: 2025-10-30
# Modified: None
# Version: 1.0
# Changelog:
#     - None

import math
from typing import Union, Optional

import torch
import torchvision
from torchvision.transforms import InterpolationMode


def imread(
        filepath: str,
        mode: Union[str, torchvision.io.ImageReadMode] = "UNCHANGED",
        size: Optional[tuple[int, int]] = None,
        **kwargs: dict,
    ) -> torch.Tensor:
    """
    Read an image from a file.

    Parameters
    ----------
    filepath : str
        Path to the image file.
    mode : str or torchvision.io.ImageReadMode, optional
        Color mode for the image. Can be any torchvision.io.ImageReadMode value.
        Default is "UNCHANGED".
    size : tuple[int, int], optional
        Desired output size as (height, width). Default is None (no resizing).
    kwargs : dict, optional
        Additional keyword arguments passed to the resize function.
    
    Returns
    -------
    torch.Tensor
        The image as a torch tensor.
    """
    img = torchvision.io.decode_image(filepath, mode=mode)

    if size is not None:
        img = imresize(img, size, **kwargs)

    return img


def imresize(
        img: torch.Tensor,
        size: tuple[int, int],
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        preserve_aspect_ratio: bool = False,
        fill: int = 0,
        antialias: bool = False,
    ) -> torch.Tensor:
    """
    Resize image possibly while maintaining aspect ratio.

    Parameters
    ----------
    img : torch.Tensor
        Input image to resize, must be a torch tensor.
    size : tuple[int, int]
        Desired output size as (height, width).
    interpolation : str, optional
        Interpolation method used for resizing, default is 'bilinear'.
    preserve_aspect_ratio : bool, optional
        Whether to preserve the aspect ratio of the image, default is False.
    fill : int, optional
        Color to use for padding if aspect ratio is preserved and padding is needed,
        default is 0.
    antialias : bool, optional
        Whether to apply an anti-aliasing filter when downsampling, default is False.
    
    Returns
    -------
    torch.Tensor
        Resized image as a torch tensor.
    """
    H, W = img.shape[-2:]

    aspect_ratio_unchanged = (H / W == size[0] / size[1])
    if preserve_aspect_ratio and not aspect_ratio_unchanged:
        scale = min(size[0] / H, size[1] / W)
        target_size = (int(H * scale), int(W * scale)) 
    else:
        target_size = size

    img = torchvision.transforms.functional.resize(
        img, size=target_size, interpolation=interpolation, antialias=antialias)

    if img.shape[-2:] != size:
        dh = size[0] - img.shape[-2]
        dw = size[1] - img.shape[-1]
        padding = (math.floor(dw / 2), math.ceil(dw / 2), 
                   math.floor(dh / 2), math.ceil(dh / 2))
        img = torch.nn.functional.pad(img, padding, mode='constant', value=fill)

    return img
