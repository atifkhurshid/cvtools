"""
Custom transforms for target variables.
"""

# Author: Atif Khurshid
# Created: 2026-08-04
# Modified: 2026-10-22
# Version: 1.1
# Changelog:
#     - 2026-08-04: Initial version.
#     - 2026-10-22: Refactored code into separate files for better organization.

from typing import Any

import torch
import torchvision.transforms.v2 as T

from .register import register_transform


@register_transform("to_int64_tensor")
class ToInt64Tensor(T.Transform):
    """
    Transform to convert a label to a tensor of type int64.
    """
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, y: Any) -> torch.Tensor:
        return torch.tensor(y, dtype=torch.int64)


@register_transform("to_float_tensor")
class ToFloatTensor(T.Transform):
    """
    Transform to convert a label to a tensor of type float32.
    """
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, y: Any) -> torch.Tensor:
        return torch.tensor(y, dtype=torch.float32)


@register_transform("to_onehot_tensor")
class ToOneHotTensor(T.Transform):
    """
    Transform to convert a label to a one-hot encoded tensor.
    """
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.n_classes = n_classes

    def __call__(self, y: Any) -> torch.Tensor:
        y_tensor = torch.tensor(y, dtype = torch.long)
        onehot_tensor = torch.zeros(self.n_classes, dtype = torch.float32)
        onehot_tensor.scatter_(0, y_tensor, 1)

        return onehot_tensor


@register_transform("squeeze_tensor")
class SqueezeTensor(T.Transform):
    """
    Transform to squeeze a tensor, removing dimensions of size 1.
    """
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        return y.squeeze()
