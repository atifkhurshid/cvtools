import cv2
import numpy as np
import matplotlib.pyplot as plt


def imread(filepath, color=cv2.COLOR_BGR2RGB, dshape=None):
    """Read, color-correct and resize image file with cv2

    Parameters
    ----------
    filepath : str
        Path to image file
    color : int, optional
        cv2 color conversion code, by default cv2.COLOR_BGR2RGB
    dshape : 2-tuple, optional
        Shape of returned image: (height, width), by default None

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

def imshow(img):
    """Display image using pyplot.imshow

    Parameters
    ----------
    img : ndarray
        Image as ndarray
    """
    plt.imshow(img)
    plt.show()