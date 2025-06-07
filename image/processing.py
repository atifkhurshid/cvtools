"""
Package: image.processing
Requirements:
    - PIL
Use: 
    - from image.processing import *
Methods:
    - resize

Author: Atif Khurshid
Created: 2022-12-18
Modified: 2025-06-08
Version: 1.2

Changelog:
    - 2025-06-08: Changed resize to imresize
    - 2025-06-08: Allowed to skip aspect ratio preservation if aspect ratio is same
    - 2025-05-22: Added fill and kwargs parameters to resize
    - 2025-05-22: Allowed resize without preserving aspect ratio
"""
import numpy as np
from PIL import Image, ImageOps


def imresize(
        img: Image.Image,
        size: tuple[int, int],
        resample: int = Image.LANCZOS,
        preserve_aspect_ratio: bool = True,
        fill: str | int | tuple = 0,
        **kwargs,
    ) -> np.ndarray:
    """Resize image possibly while maintianing aspect ratio

    Args:
        img: PIL.Image or ndarray
            Image to be resized
        size: tuple
            Size after resize operation, (height, width)
        resample: PIL.Image filter, optional
            Resampling filter, default is Image.LANCZOS
        preserve_aspect_ratio: bool, optional
            If True, image will be resized to fit within the given size
            while maintaining the aspect ratio, filling any empty regions
            with padding.
        fill: str or int or tuple, optional
            Color to use for padding, default is 0 (black)
        kwargs: optional
            Additional arguments to be passed to the PIL.Image.resize() or thumbnail() methods

    Returns:
        img: PIL.Image
            Resized image as PIL.Image
    """
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    else:
        img = img.copy()

    size = size[::-1]  # PIL uses (width, height) format

    same_aspect_ratio = img.width / img.height == size[0] / size[1]
    if not same_aspect_ratio and preserve_aspect_ratio:
        img.thumbnail(size, resample=resample, **kwargs)
        # Pad image if needed
        # Source: https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
        if img.size != size:
            dw = size[0] - img.width
            dh = size[1] - img.height
            padding = (dw//2, dh//2, dw-(dw//2), dh-(dh//2))
            img = ImageOps.expand(img, border=padding, fill=fill)
    else:
        img = img.resize(size, resample=resample, **kwargs)

    return np.asarray(img)
