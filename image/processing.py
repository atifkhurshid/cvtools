"""
Package: image.processing
Requirements:
    - PIL
Use: 
    - from cvtools.image.processing import *
Methods:
    - resize
"""
from typing import Tuple

from PIL import Image, ImageOps


def resize(
        img: Image.Image,
        size: Tuple[int, int],
        resample: int = Image.LANCZOS
    ) -> Image.Image:
    """Resize image while maintianing aspect ratio

    Args:
        img: PIL.Image or ndarray
            Image to be resized
        size: tuple
            Size after resize operation, (height, width)
        resample: PIL.Image filter, optional
            Resampling filter, default is Image.LANCZOS

    Returns:
        img: PIL.Image
            Resized image as PIL.Image
    """
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    else:
        img = img.copy()

    # PIL expects size as (x, y) instead of (h, w)
    size = size[::-1]

    img.thumbnail(size, resample=resample)

    # Pad image if needed
    # Source: https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    if img.size != size:
        dw = size[0] - img.size[0]
        dh = size[1] - img.size[1]
        padding = (dw//2, dh//2, dw-(dw//2), dh-(dh//2))
        img = ImageOps.expand(img, padding)

    return img
