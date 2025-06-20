"""
Image I/O module.
"""

# Author: Atif Khurshid
# Created: 2022-12-16
# Modified: 2025-05-23
# Version: 2.2
# Changelog:
#     - 2025-05-23: Changed imread to use processing.resize
#     - 2025-05-22: Change size parameter ordering to (height, width)
#     - 2025-05-22: Added resampling parameter to imread()
#     - 2025-05-22: Change image processing library from OpenCV to PIL

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .processing import imresize as resize


def imread(
        filepath: str,
        mode: str | None=None,
        size: tuple[int, int] | None=None,
        **kwargs,
    ) -> np.ndarray:
    """
    Read image using PIL and convert to ndarray.
    Image can optionaly be resized and converted to a specific color mode.

    Parameters
    ----------
    filepath : str
        Path to image file
    mode : int, optional
        PIL color conversion mode, default is No Conversion
    size : 2-tuple, optional
        Shape of returned image: (height, width), default is None
    kwargs : dict, optional
        Additional arguments passed to resize function, such as resample method.

    Returns
    -------
    ndarray
        Image as ndarray, None if file cannot be read.
    
    Examples
    --------
    >>> from PIL import Image
    >>> from cvtools.image import imread
    >>> img = imread('path/to/image.jpg')
    >>> img = imread('path/to/image.jpg', mode='RGB', size=(256, 256), resample=Image.BICUBIC)
    >>> img = imread('path/to/image.jpg', size=(256, 256), preserve_aspect_ratio=False)
    """
    with Image.open(filepath) as img:
        if mode:
            img = img.convert(mode)
        if size:
            img = resize(img, size, **kwargs)
        img = np.asarray(img, copy=True)

    return img


def imwrite(
        filepath: str,
        img: np.ndarray,
        mode: str | None=None,
        **params
    ):
    """
    Write image to file using PIL.

    Parameters
    ----------
    filepath : str
        Path to save the image file
    img : ndarray
        Image as ndarray
    mode : str, optional
        PIL color conversion mode, default is No Conversion
    params : dict, optional
        Additional parameters passed to PIL.Image.save(), such as quality for JPEG.

    Examples
    --------
    >>> from cvtools.image import imwrite
    >>> imwrite('path/to/save/image.jpg', img)
    >>> imwrite('path/to/save/image.jpg', img, mode='RGB', quality=95)
    """
    img = Image.fromarray(img)
    if mode:
        img = img.convert(mode)
    img.save(filepath, **params)


def imshow(
        img: np.ndarray,
        axis: bool=False,
        block: bool=False,
        **kwargs,
    ) -> None:
    """
    Display image using matplotlib.

    Parameters
    ----------
    img : ndarray
        Image as ndarray
    axis: bool, optional
        Whether to display axis, default is False
    block: bool, optional
        Whether to wait for all figures to be closed before returning, default is False
    kwargs: dict, optional
        Additional arguments passed to matplotlib.pyplot.imshow()
    
    Examples
    --------
    >>> from cvtools.image import imshow
    >>> imshow(img)
    >>> imshow(img, axis=True, block=True, cmap='gray')
    """
    # Set default cmap for grayscale images to gray
    if (img.ndim == 2 or img.shape[2] == 1) and "cmap" not in kwargs:
        kwargs["cmap"] = "gray"
    plt.imshow(img, **kwargs)
    if not axis:
        plt.axis("off")
    plt.show(block=block)
