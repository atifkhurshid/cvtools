"""
ArcFace Loss implememntation in PyTorch.

Adapted from https://github.com/Vision-At-SEECS/streamface/blob/main/experiments/cnn/arcface/utils.py
"""

# Author: Atif Khurshid
# Created: 2026-05-06
# Modified: None
# Version: 1.0
# Changelog:
#    - 2026-05-06: Initial implementation.

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcLayer(nn.Module):

    def __init__(
            self,
            embedding_size: int,
            n_classes: int
        ):
        """
        ArcLayer for additive angular margin loss.

        Parameters
        ----------
        embedding_size : int
            Dimensionality of the input embedding space.
        n_classes : int
            Number of classes for classification.

        Attributes
        ----------
        weights : torch.nn.Parameter
            Learnable parameters representing the class weights in the embedding space.
        """
        super().__init__()

        self.weights = nn.Parameter(torch.randn(embedding_size, n_classes))

        nn.init.kaiming_normal_(self.weights)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ArcLayer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, embedding_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_classes).
        """
        weights = F.normalize(self.weights, p=2, dim=0)

        return torch.mm(x, weights)


class ArcLoss(nn.Module):

    def __init__(
            self,
            n_classes: int,
            margin: float = 0.5,
            scale: float = 64,
            **kwargs,
        ):
        """
        Additive angular margin loss.

        Original implementation: https://github.com/luckycallor/InsightFace-tensorflow

        Parameters
        ----------
        n_classes : int
            Number of classes for classification.
        margin : float, optional
            Angular margin in radians. Default is 0.5.
        scale : float, optional
            Scaling factor for the logits. Default is 64.
        **kwargs : dict
            Additional keyword arguments for the underlying cross-entropy loss function.
        """
        super().__init__()

        self.n_classes = n_classes
        self.margin = torch.tensor(margin)
        self.scale = scale
        self.threshold = torch.cos(torch.pi - self.margin)
        self.cos_m = torch.cos(self.margin)
        self.sin_m = torch.sin(self.margin)

        # Safe margin: https://github.com/deepinsight/insightlabelsface/issues/108
        self.safe_margin = self.sin_m * self.margin

        self.ce = nn.CrossEntropyLoss(**kwargs)


    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor
        ) -> torch.Tensor:
        """
        Compute the ArcFace loss.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor of shape (batch_size, n_classes) 
        target : torch.Tensor
            Target labels of shape (batch_size,).
        """
        # Calculate the cosine value of theta + margin.
        cos_t = input
        sin_t = torch.sqrt(1 - torch.square(cos_t))

        cos_t_margin = torch.where(cos_t > self.threshold,
                                    cos_t * self.cos_m - sin_t * self.sin_m,
                                    cos_t - self.safe_margin)

        mask = F.one_hot(target.to(torch.int64), num_classes=self.n_classes).float()

        cos_t_onehot = cos_t * mask
        cos_t_margin_onehot = cos_t_margin * mask

        # Calculate the final scaled logits.
        logits = (cos_t + cos_t_margin_onehot - cos_t_onehot) * self.scale

        loss = self.ce(logits, target)

        return loss