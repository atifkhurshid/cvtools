"""
Center Loss implememntation in PyTorch.
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


class CenterLoss(nn.Module):
    
    def __init__(
            self,
            n_classes: int,
            embedding_dim: int,
            lambda_: float = 1.0
        ):
        """
        Center Loss for deep feature learning.

        Parameters
        ----------
        n_classes : int
            Number of classes in the dataset.
        embedding_dim : int
            Dimensionality of the embedding space.
        lambda_ : float, optional
            Weighting factor for the center loss. Default is 1.0.
        
        Attributes
        ----------
        centers : torch.nn.Parameter
            Learnable parameters representing the class centers in the embedding space.
        """
        super().__init__()

        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        self.lambda_ = lambda_

        self.centers = nn.Parameter(torch.randn(n_classes, embedding_dim))


    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor
        ) -> torch.Tensor:
        """
        Compute the center loss.

        Parameters
        ----------
        input : torch.Tensor
            Input features.
        target : torch.Tensor
            Target labels.

        Returns
        -------
        torch.Tensor
            Computed center loss.
        """
        centers_batch = self.centers[target]
        loss = F.mse_loss(input, centers_batch)

        return self.lambda_ * loss
