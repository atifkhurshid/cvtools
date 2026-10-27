"""
Registry for torchvision image transforms.
"""

# Author: Atif Khurshid
# Created: 2026-08-04
# Modified: 2026-10-22
# Version: 1.1
# Changelog:
#     - 2026-08-04: Initial version.
#     - 2026-10-22: Refactored code into separate files for better organization.


from typing import Callable, Any, Optional
from dataclasses import dataclass, field

import torchvision.transforms.v2 as T


TRANSFORM_REGISTRY: dict[str, Callable] = {}

@dataclass
class TransformSpec:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


def register_transform(name: str) -> Callable:
    """
    Register a new transform class in the TRANSFORM_REGISTRY.

    Parameters
    ----------
    name : str
        Name of the transform to register.
    transform_class : Callable
        The transform class to register.
    """
    def wrapper(transform_class: Callable):
        if name in TRANSFORM_REGISTRY:
            raise ValueError(f"Transform '{name}' is already registered.")
        TRANSFORM_REGISTRY[name] = transform_class
    return wrapper


def build_transforms(specs: list[TransformSpec]) -> Optional[T.Compose]:

    if len(specs):
        transforms = T.Compose([TRANSFORM_REGISTRY[s.name](**s.kwargs) for s in specs])
    else:
        transforms = None

    return transforms

