"""
Direct classification from pixel values.
"""

# Author: Atif Khurshid
# Created: 2026-10-09
# Modified: None
# Version: 1.0
# Changelog:
#     - 2026-10-09: Initial version.

from typing import Optional

import torch
import torch.nn as nn

from .cnn import PyTorchCNNModel


class PixelClassifier(PyTorchCNNModel):

    def __init__(
        self,
        in_channels: int,
        avgpool_size: tuple[int, int] = (32, 32),
        classifier: Optional[str] = "linear",
        n_classes: int = 1000,
        classifier_hidden_dim: int = 4096,
        classifier_dropout: float = 0.5,
    ):
        """
        Image classifier that takes raw pixel values as input.

        Parameters
        ----------
        in_channels : int
            Number of input channels (e.g., 3 for RGB images).
        avgpool_size : tuple[int, int], optional
            Size of the adaptive average pooling layer.
        classifier : Optional[str], optional
            Type of classifier to use. Options are "linear", "mlp", "mlp2", or "arcface".
            If None, no classifier is applied.
        n_classes : int, optional
            Number of classes in the classification task.
        classifier_hidden_dim : int, optional
            Hidden dimension for the MLP classifiers. Ignored if classifier is not "mlp"
            or "mlp2".
        classifier_dropout : float, optional
            Dropout rate for the MLP classifiers. Ignored if classifier is not "mlp" or "mlp2".
        """
        super().__init__(
            avgpool_size = avgpool_size,
            feature_depth = in_channels,
            classifier = classifier,
            n_classes = n_classes,
            classifier_hidden_dim = classifier_hidden_dim,
            classifier_dropout = classifier_dropout,
        )

        self.features = nn.Identity()  # No feature extractor, input is in pixel space
