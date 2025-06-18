"""
Image processing module.
"""

# Author: Atif Khurshid
# Created: 2022-12-18
# Modified: 2025-06-08
# Version: 1.2
# Changelog:
#     - 2025-06-08: Changed resize to imresize
#     - 2025-06-08: Allowed to skip aspect ratio preservation if aspect ratio is same
#     - 2025-05-22: Added fill and kwargs parameters to resize
#     - 2025-05-22: Allowed resize without preserving aspect ratio

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
    """
    Resize image possibly while maintaining aspect ratio.

    Parameters
    ----------
    img : Image.Image or ndarray
        Input image to resize, can be a PIL Image or a numpy array.
    size : tuple[int, int]
        Desired output size as (height, width).
    resample : int, optional
        Resampling filter to use when resizing, default is Image.LANCZOS.
    preserve_aspect_ratio : bool, optional
        Whether to preserve the aspect ratio of the image, default is True.
    fill : str, int, or tuple, optional
        Color to use for padding if aspect ratio is preserved and padding is needed,
        default is 0 (black). Can be a string (e.g., 'white'), an integer (grayscale),
        or a tuple for RGB/RGBA colors.
    **kwargs : dict, optional
        Additional keyword arguments passed to the PIL resize method.

    Returns
    -------
    np.ndarray
        Resized image as a numpy array.

    Examples
    --------
    >>> from PIL import Image
    >>> from cvtools.image import imresize
    >>> img = Image.open('path/to/image.jpg')
    >>> resized_img = imresize(img, (256, 256), resample=Image.BICUBIC)
    >>> resized_img = imresize(img, (256, 256), preserve_aspect_ratio=False)
    >>> resized_img = imresize(img, (256, 256), fill='white')
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
