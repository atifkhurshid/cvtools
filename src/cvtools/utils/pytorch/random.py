"""
Utility functions for random number generation.
"""

# Author: Atif Khurshid
# Created: 2025-11-11
# Modified: 2026-05-11
# Version: 1.1
# Changelog:
#     - 2025-11-11: Added set_seed function.
#     - 2026-05-11: Updated set_seed function to include torch.device type.

import random

import torch
import numpy as np


def set_seed(seed: int, device: torch.device = torch.device("cpu")) -> None:
    """
    Set the random seed for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to set.
    device : torch.device, optional
        The device type ("cpu" or "cuda"), by default "cpu".
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if device == torch.device("cuda"):
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
