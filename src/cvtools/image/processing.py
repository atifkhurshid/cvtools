"""
Image processing module.
"""

# Author: Atif Khurshid
# Created: 2022-12-18
# Modified: 2026-03-04
# Version: 1.7
# Changelog:
#     - 2025-05-22: Added fill and kwargs parameters to resize
#     - 2025-05-22: Allowed resize without preserving aspect ratio
#     - 2025-06-08: Changed resize to imresize
#     - 2025-06-08: Allowed to skip aspect ratio preservation if aspect ratio is same
#     - 2025-10-29: Changed image processing library back to OpenCV
#     - 2026-03-03: Added imresize_maximum function
#     - 2026-03-03: Added imscale function
#     - 2026-03-04: Refactored image padding into a separate function.

from math import ceil, floor

import cv2
import numpy as np


def imscale(
        img: np.ndarray,
        scale: float,
        interpolation: int = cv2.INTER_AREA,
    ) -> np.ndarray:
    """
    Scale image by a given factor.

    Parameters
    ----------
    img : ndarray
        Input image to scale, must be a numpy array.
    scale : float
        Scaling factor to apply to the image.
    interpolation : int, optional
        Interpolation method used for scaling, default is cv2.INTER_AREA.

    Returns
    -------
    np.ndarray
        Scaled image as a numpy array.

    Examples
    --------
    >>> from cvtools.image import imscale
    >>> import numpy as np
    >>> img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    >>> scaled_img = imscale(img, 0.5)
    """
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale)) # (width, height)
    img = cv2.resize(img, new_size, interpolation=interpolation)
    
    return img


def imresize(
        img: np.ndarray,
        size: tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
        preserve_aspect_ratio: bool = False,
        **kwargs: dict,
    ) -> np.ndarray:
    """
    Resize image possibly while maintaining aspect ratio.

    Parameters
    ----------
    img : ndarray
        Input image to resize, must be a numpy array.
    size : tuple[int, int]
        Desired output size as (height, width).
    interpolation : int, optional
        Interpolation method used for resizing, default is cv2.INTER_LINEAR.
    preserve_aspect_ratio : bool, optional
        Whether to preserve the aspect ratio of the image, default is False.
    **kwargs : dict
        Additional keyword arguments to pass to np.pad when preserving aspect ratio.

    Returns
    -------
    np.ndarray
        Resized image as a numpy array.

    Examples
    --------
    >>> from cvtools.image import imresize
    >>> import numpy as np
    >>> img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    >>> resized_img = imresize(img, (200, 200), preserve_aspect_ratio=True, mode='constant', constant_values=255)
    """
    H, W = img.shape[:2]
    aspect_ratio_unchanged = (H / W == size[0] / size[1])

    if preserve_aspect_ratio and not aspect_ratio_unchanged:
        scale = min(size[0] / H, size[1] / W)
        new_size = (int(W * scale), int(H * scale)) # (width, height)
        img = cv2.resize(img, new_size, interpolation=interpolation)
        img = _pad_image_to_size(img, size, **kwargs)
    else:
        # OpenCV uses (width, height) format
        img = cv2.resize(img, size[::-1], interpolation=interpolation)

    return img


def imresize_maximum(
        img: np.ndarray,
        max_size: int,
        interpolation: int = cv2.INTER_AREA,
    ) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio such that the longest side is at most max_size.

    Parameters
    ----------
    img : ndarray
        Input image to resize, must be a numpy array.
    max_size : int
        Desired maximum size of the longest side of the output image.
    interpolation : int, optional
        Interpolation method used for resizing, default is cv2.INTER_AREA.

    Returns
    -------
    np.ndarray
        Resized image as a numpy array.

    Examples
    --------
    >>> from cvtools.image import imresize_maximum
    >>> import numpy as np
    >>> img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    >>> resized_img = imresize_maximum(img, 200)
    """
    H, W = img.shape[:2]
    
    if max(H, W) > max_size:
        scale = max_size / max(H, W)
        new_size = (int(W * scale), int(H * scale)) # (width, height)
        img = cv2.resize(img, new_size, interpolation=interpolation)

    return img


def _pad_image_to_size(
        img: np.ndarray,
        size: tuple[int, int],
        **kwargs: dict,
    ) -> np.ndarray:
        """
        Pad the image to the specified size if it is smaller.

        Parameters
        ----------
        img : ndarray
            Input image to pad, must be a numpy array.
        size : tuple[int, int]
            Desired output size as (height, width).
        **kwargs : dict
            Additional keyword arguments to pass to np.pad.

        Returns
        -------
        np.ndarray
            Padded image as a numpy array.    
        """
        if img.shape[:2] != size:
            dh = size[0] - img.shape[0]
            dw = size[1] - img.shape[1]
            padding = [(floor(dh / 2), ceil(dh / 2)), (floor(dw / 2), ceil(dw / 2))]
            padding.extend([(0, 0)] * (img.ndim - 2))  # No padding for channels
            img = np.pad(img, padding, **kwargs)
        
        return img
