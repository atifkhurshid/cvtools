"""
Numpy array utilities.
"""

# Author: Atif Khurshid
# Created: 2025-07-04
# Modified: None
# Version: 1.0
# Changelog:
#     - None

from collections import defaultdict

import numpy as np


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
    """
    
    groups_dict = defaultdict(list)
    for arr in arrays:
        groups_dict[arr.shape].append(arr)
    
    groups = []
    for group in groups_dict.values():
        group = np.array(group)
        groups.append(group)
    
    return groups
