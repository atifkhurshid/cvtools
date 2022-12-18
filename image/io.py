"""
Package: image.io
Requirements:
    - cv2
    - matplotlib
Use: 
    - from cvtools.image.io import *
Methods:
    - imread
    - imwrite
    - imshow
"""
from typing import Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


def imread(
        filepath: str,
        color: int = cv2.COLOR_BGR2RGB,
        dshape: Tuple[int, int] = None
    ) -> np.ndarray:
    """Read, color-correct and resize image file with cv2

    Parameters
    ----------
    filepath : str
        Path to image file
    color : int, optional
        cv2 color conversion code, default is cv2.COLOR_BGR2RGB
    dshape : 2-tuple, optional
        Shape of returned image: (height, width), default is None

    Returns
    -------
    ndarray
        Image as ndarray, None if file cannot be read.
    """
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is not None:
        if color:
            img = cv2.cvtColor(img, color)
        if dshape:
            # INTER_AREA interpolation when shrinking, INTER_LINEAR when enlarging
            magnification = (dshape[0] * dshape[1]) / (img.shape[0] * img.shape[1])
            if magnification < 1:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_LINEAR
            # cv2 expects size as (x, y) instead of (h, w) in dshape
            dsize = dshape[::-1]
            img = cv2.resize(img, dsize, interpolation=interpolation)

    return img


def imwrite(
        filepath: str,
        img: np.ndarray,
        color: int = cv2.COLOR_RGB2BGR,
        **kwargs: dict,
    ) -> bool:
    """Save image using cv2

    Args:
        filepath: str
            Path to save image file
        img: ndarray
            Image as ndarray
        color: int, optional
            cv2 color conversion code, default is cv2.COLOR_RGB2BGR
        kwargs: dict, optional
            Additional arguments passed to cv2.imwrite()

    Returns:
        res: bool
            Success status of saving operation
    """
    if color:
        img = cv2.cvtColor(img, color)
    res = cv2.imwrite(filepath, img, **kwargs)

    return res


def imshow(
        img: np.ndarray,
        axis: bool = False,
        block: bool = False,
        **kwargs: dict,
    ) -> None:
    """Display image using pyplot.imshow

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
    plt.imshow(img, **kwargs)
    if not axis:
        plt.axis("off")
    plt.show(block=block)
