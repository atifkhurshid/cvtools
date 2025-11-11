"""
Utility functions for random number generation.
"""

# Author: Atif Khurshid
# Created: 2025-11-11
# Modified: None
# Version: 1.0
# Changelog:
#     - 2025-11-11: Added set_seed function.

import random

import torch
import numpy as np


def set_seed(seed: int, device: str = "cpu"):
    """
    Set the random seed for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to set.
    device : str, optional
        The device type ("cpu" or "cuda"), by default "cpu".
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
