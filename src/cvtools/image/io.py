"""
Package: image.io
Requirements:
    - numpy
    - matplotlib
    - pillow
Use: 
    - from image.io import *
Methods:
    - imread
    - imwrite
    - imshow

Author: Atif Khurshid
Created: 2022-12-16
Modified: 2025-05-23
Version: 2.2

Changelog:
    - 2025-05-23: Changed imread to use processing.resize
    - 2025-05-22: Change size parameter ordering to (height, width)
    - 2025-05-22: Added resampling parameter to imread()
    - 2025-05-22: Change image processing library from OpenCV to PIL
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .processing import imresize


def imread(
        filepath: str,
        mode: str = None,
        size: tuple[int, int] = None,
        **kwargs,
    ) -> np.ndarray:
    """Read, color-correct and resize image file with PIL

    Parameters
    ----------
    filepath : str
        Path to image file
    mode : int, optional
        PIL color conversion mode, default is No Conversion
    size : 2-tuple, optional
        Shape of returned image: (height, width), default is None
    resample : int, optional
        Resampling filter for resize, default is Image.LANCZOS

    Returns
    -------
    ndarray
        Image as ndarray, None if file cannot be read.
    """
    with Image.open(filepath) as img:
        if mode:
            img = img.convert(mode)
        if size:
            img = imresize(img, size, **kwargs)
        img = np.asarray(img, copy=True)

    return img


def imwrite(
        filepath: str,
        img: np.ndarray,
        mode: str = None,
        **params
    ) -> None:
    """Save image using PIL

    Args:
        filepath: str
            Path to save image file
        img: ndarray
            Image as ndarray
        color: int, optional
            PIL color conversion mode, default is No Conversion
        params: dict, optional
            Additional arguments passed to Image.save()
    """
    img = Image.fromarray(img)
    if mode:
        img = img.convert(mode)
    img.save(filepath, **params)


def imshow(
        img: np.ndarray,
        axis: bool = False,
        block: bool = False,
        **kwargs,
    ) -> None:
    """Display image using matplotlib

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
    """
    # Set default cmap for grayscale images to gray
    if (img.ndim == 2 or img.shape[2] == 1) and "cmap" not in kwargs:
        kwargs["cmap"] = "gray"
    plt.imshow(img, **kwargs)
    if not axis:
        plt.axis("off")
    plt.show(block=block)
