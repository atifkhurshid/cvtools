"""
Image processing module.
"""

# Author: Atif Khurshid
# Created: 2022-12-18
# Modified: 2025-10-29
# Version: 1.5
# Changelog:
#     - 2025-05-22: Added fill and kwargs parameters to resize
#     - 2025-05-22: Allowed resize without preserving aspect ratio
#     - 2025-06-08: Changed resize to imresize
#     - 2025-06-08: Allowed to skip aspect ratio preservation if aspect ratio is same
#     - 2025-10-29: Changed image processing library back to OpenCV

import cv2
import numpy as np


def imresize(
        img: np.ndarray,
        size: tuple[int, int],
        interpolation: int = cv2.INTER_LINEAR,
        preserve_aspect_ratio: bool = False,
        fill: int = 0,
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
    fill : int, optional
        Color to use for padding if aspect ratio is preserved and padding is needed,
        default is 0.
    **kwargs : dict, optional
        Additional keyword arguments passed to the PIL resize method.

    Returns
    -------
    np.ndarray
        Resized image as a numpy array.

    Examples
    --------
    >>> from cvtools.image import imresize
    >>> import numpy as np
    >>> img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    >>> resized_img = imresize(img, (200, 200), preserve_aspect_ratio=True, fill=255)
    """
    H, W = img.shape[:2]

    aspect_ratio_unchanged = (H / W == size[0] / size[1])
    if preserve_aspect_ratio and not aspect_ratio_unchanged:
        
        scale = min(size[0] / H, size[1] / W)
        new_size = (int(W * scale), int(H * scale)) # (width, height)

        img = cv2.resize(img, new_size, interpolation=interpolation)

        if img.shape[:2] != size:
            dh = size[0] - img.shape[0]
            dw = size[1] - img.shape[1]
            padding = [(dh // 2, dh // 2), (dw // 2, dw // 2)]
            padding.extend([(0, 0)] * (img.ndim - 2))  # No padding for channels
            img = np.pad(img, padding, mode='constant', constant_values=fill)

    # OpenCV uses (width, height) format
    img = cv2.resize(img, size[::-1], interpolation=interpolation)

    return img
