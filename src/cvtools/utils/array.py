"""
Numpy array utilities.
"""

# Author: Atif Khurshid
# Created: 2025-07-04
# Modified: 2026-02-26
# Version: 1.1
# Changelog:
#     - 2025-08-27: Updated padding logic to allow specific sizes for padding.
#     - 2026-02-26: Added pad_array_to_shape function.

from typing import Union
from collections import defaultdict

import numpy as np


def pad_array_to_shape(
        array: np.ndarray,
        target_shape: tuple,
        **kwargs: dict,
    ) -> np.ndarray:
    """
    Pad a numpy array to a target shape.

    Parameters
    ----------
    array : np.ndarray
        Input array to be padded.
    target_shape : tuple of int
        Desired shape of the output array after padding.
    **kwargs : dict
        Additional keyword arguments for the padding function.

    Returns
    -------
    np.ndarray
        Padded array with the specified target shape.

    Examples
    ---------
    >>> arr = np.array([[1, 2], [3, 4]])
    >>> padded = pad_array_to_shape(arr, (4, 4), mode='constant', constant_values=0)
    >>> print(padded)
    [[0 0 0 0]
     [0 1 2 0]
     [0 3 4 0]
     [0 0 0 0]]
    """
    shape = array.shape
    padding_offsets = [target_shape[i] - shape[i] for i in range(len(shape))]
    padding = tuple([(p // 2, p - p // 2) for p in padding_offsets])

    padded_array = np.pad(
        array,
        padding,
        **kwargs,
    )

    return padded_array


def group_arrays_by_shape(
        arrays: list[np.ndarray]
    ) -> list[np.ndarray]:
    """
    Group a list of numpy arrays by their shapes.

    Parameters
    ----------
    arrays : list of np.ndarray
        List of numpy arrays to be grouped.

    Returns
    -------
    list of np.ndarray
        List of numpy arrays, where each array contains all arrays of the same shape.
    
    Examples
    ---------
    >>> arr1 = np.array([[1, 2], [3, 4]])
    >>> arr2 = np.array([[5, 6, 7], [8, 9, 10]])
    >>> arr3 = np.array([[11, 12], [13, 14]])
    >>> grouped = group_arrays_by_shape([arr1, arr2, arr3])
    >>> print([g.shape for g in grouped])
    [(2, 2), (2, 3)]
    """
    groups_dict = defaultdict(list)
    for arr in arrays:
        groups_dict[arr.shape].append(arr)
    
    groups = []
    for group in groups_dict.values():
        group = np.array(group, dtype=group[0].dtype)
        groups.append(group)
    
    return groups


def pad_arrays_to_uniform_size(
        arrays: list[np.ndarray],
        size: Union[str, tuple[int, int]] = "auto",
        mode: str = "constant",
        **kwargs: dict,
    ) -> np.ndarray:
    """
    Pads a list of 2D numpy arrays to the size of the largest array in the list.

    Parameters
    ----------
    arrays : list of np.ndarray
        List of 2D numpy arrays to be padded.
    size : str or tuple of int, optional
        Target size for the padded arrays. Default is "auto", which means the size will be determined
        by the largest array in the list. If a tuple is provided, it should be in the form
        (height, width).
    mode : str, optional
        Padding mode. Default is "constant".
    **kwargs : dict, optional
        Additional keyword arguments for the padding function.
        
    Returns
    -------
    np.ndarray
        A 3D numpy array, containing input arrays of the same size.
    
    Examples
    ---------
    >>> arr1 = np.array([[1, 2], [3, 4]])
    >>> arr2 = np.array([[5, 6, 7], [8, 9, 10]])
    >>> padded = pad_arrays_to_uniform_size([arr1, arr2],  size="auto", mode='constant', constant_values=0)
    >>> print(padded)
    [[[ 1  2  0]
      [ 3  4  0]],
     [[ 5  6  7]
      [ 8  9 10]]]
    """
    if size == "auto":
        max_rows = max([arr.shape[0] for arr in arrays])
        max_cols = max([arr.shape[1] for arr in arrays])
    else:
        max_rows, max_cols = size

    padded_arrays = []
    for arr in arrays:
        rows, cols = arr.shape
        pad_top = int(np.ceil((max_rows - rows) / 2))
        pad_bottom = int(np.floor((max_rows - rows) / 2))
        pad_left = int(np.ceil((max_cols - cols) / 2))
        pad_right = int(np.floor((max_cols - cols) / 2))

        padded = np.pad(arr, ((pad_top, pad_bottom), (pad_left, pad_right)), mode=mode, **kwargs)
        padded_arrays.append(padded)

    return np.array(padded_arrays, dtype=padded_arrays[0].dtype)
