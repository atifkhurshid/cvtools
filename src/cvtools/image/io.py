"""
Image I/O module.
"""

# Author: Atif Khurshid
# Created: 2022-12-16
# Modified: 2025-10-29
# Version: 2.5
# Changelog:
#     - 2025-05-22: Change image processing library from OpenCV to PIL
#     - 2025-05-22: Change size parameter ordering to (height, width)
#     - 2025-05-22: Added resampling parameter to imread()
#     - 2025-05-23: Changed imread to use processing.resize
#     - 2025-10-29: Changed image processing library back to OpenCV

import cv2
import numpy as np
import matplotlib.pyplot as plt

from .processing import imresize as resize


def imread(
        filepath: str,
        mode: str | int | None = None,
        size: tuple[int, int] | None = None,
        **kwargs,
    ) -> np.ndarray:
    """
    Read image using OpenCV.
    Image can optionaly be resized and converted to a specific color mode.

    Parameters
    ----------
    filepath : str
        Path to image file
    mode : str or int, optional
        Color conversion mode. Can be one of "RGB", "GRAY" or any cv2 color conversion code.
        Default is None (no conversion).
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
    >>> from cvtools.image import imread
    >>> img = imread('path/to/image.jpg')
    >>> img_rgb = imread('path/to/image.jpg', mode='RGB')
    >>> img_gray = imread('path/to/image.jpg', mode='GRAY', size=(256, 256))
    """
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    
    if mode is not None:
        if mode == 'RGB':
            mode = cv2.COLOR_BGR2RGB
        elif mode == 'GRAY':
            mode = cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, mode)

    if size is not None:
        img = resize(img, size, **kwargs)

    return img


def imwrite(
        filepath: str,
        img: np.ndarray,
        mode: str | None = None,
    ):
    """
    Write image to file using OpenCV.

    Parameters
    ----------
    filepath : str
        Path to save the image file
    img : ndarray
        Image as ndarray
    mode : int, optional
        OpenCV color conversion mode for converting non-standard images
        (e.g., with alpha channel) to BGR-like before saving. RGB images
        are automatically converted to BGR. Default is None.

    Examples
    --------
    >>> from cvtools.image import imwrite
    >>> imwrite('path/to/save/image.jpg', img))
    """
    if img.ndim == 2:
        pass  # Grayscale image, no conversion needed
    else:
        if img.ndim == 3 and img.shape[2] == 3:
            if mode is None:
                mode = cv2.COLOR_RGB2BGR
        else:
            assert mode is not None, \
                "X2BGR conversion mode must be specified for non-RGB images"
        img = cv2.cvtColor(img, mode)
    
    cv2.imwrite(filepath, img)


def imshow(
        img: np.ndarray,
        axis: bool = False,
        block: bool = False,
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
