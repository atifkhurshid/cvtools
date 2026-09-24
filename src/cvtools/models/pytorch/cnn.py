"""
Base class for CNN models.
"""

# Author: Atif Khurshid
# Created: 2025-11-11
# Modified: 2026-09-24
# Version: 1.3
# Changelog:
#     - 2025-11-11: Initial version.
#     - 2025-11-11: Changed init weights to allow submodule initialization only.
#     - 2026-09-24: Moved hooks and weight activation to the base class.
#     - 2026-09-24: Added default classifiers.

from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn

from .base import PyTorchModel
from ...layers.pytorch import L2Norm
from ...losses.pytorch import ArcLayer


class PyTorchCNNModel(PyTorchModel):
    
    def __init__(
            self,
            avgpool_size: Optional[tuple[int, int]] = (1, 1),
            feature_depth: int = 512,
            classifier: Optional[str] = "linear",
            n_classes: Optional[int] = None,
            classifier_hidden_dim: int = 4096,
            classifier_dropout: float = 0.5,
        ):
        """
        Base class for CNN models.

        Parameters
        ----------
        avgpool_size : Optional[tuple[int, int]], optional
            Size of the adaptive average pooling layer. If None, no pooling is applied.
        feature_depth : int, optional
            Depth of the feature maps. Default is 512.
        classifier : Optional[str], optional
            Type of classifier to use. Options are "linear", "mlp", "mlp2", or "arcface". If None, no classifier is applied.
        n_classes : Optional[int], optional
            Number of classes in the classification task.
        classifier_hidden_dim : int, optional
            Hidden dimension for the MLP classifiers. Ignored if classifier is not "mlp" or "mlp2".
        classifier_dropout : float, optional
            Dropout rate for the MLP classifiers. Ignored if classifier is not "mlp" or "mlp2".
        """
        super().__init__()

        self.features: nn.Module

        self.avgpool = nn.Identity()
        self.classifier = nn.Identity()

        embedding_dim = feature_depth

        if avgpool_size is not None:
            self.avgpool = nn.AdaptiveAvgPool2d(avgpool_size)
            embedding_dim = embedding_dim * avgpool_size[0] * avgpool_size[1]

        if classifier is not None:

            assert classifier in ["linear", "mlp", "mlp2", "arcface"], \
                f"Unsupported classifier: {classifier}. Supported classifiers are 'linear', 'mlp', 'mlp2', and 'arcface'."
            
            assert n_classes is not None, \
                "n_classes must be specified if classifier is not None."

            classifier_modules = OrderedDict([])
            classifier_modules["flatten"] = nn.Flatten()

            if classifier == "arcface":
                classifier_modules["l2_norm"] = L2Norm()
                classifier_modules["arcface"] = ArcLayer(embedding_dim, n_classes)

            else:
                fc = 1
                if "mlp" in classifier:

                    assert classifier_hidden_dim is not None, \
                        "classifier_hidden_dim must be specified for MLP classifiers."

                    classifier_modules[f"fc{fc}"] = nn.Linear(embedding_dim, classifier_hidden_dim)
                    classifier_modules[f"relu{fc}"] = nn.ReLU()
                    if classifier_dropout > 0:
                        classifier_modules[f"dropout{fc}"] = nn.Dropout(classifier_dropout)
                    fc += 1
                    embedding_dim = classifier_hidden_dim

                    if classifier == "mlp2":
                        classifier_modules[f"fc{fc}"] = nn.Linear(classifier_hidden_dim, classifier_hidden_dim)
                        classifier_modules[f"relu{fc}"] = nn.ReLU()
                        if classifier_dropout > 0:
                            classifier_modules[f"dropout{fc}"] = nn.Dropout(classifier_dropout)
                        fc += 1

                classifier_modules[f"fc{fc}"] = nn.Linear(embedding_dim, n_classes)
            
            self.classifier = nn.Sequential(classifier_modules)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.

        The input is sequentially passed through the feature extractor,
        average pooling, and classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)

        return x


    def freeze_backbone(self):
        """
        Freeze the weights of the backbone feature extractor.
        """
        for param in self.features.parameters():
            param.requires_grad = False


    def unfreeze_backbone(self):
        """
        Unfreeze the weights of the backbone feature extractor.
        """
        for param in self.features.parameters():
            param.requires_grad = True
