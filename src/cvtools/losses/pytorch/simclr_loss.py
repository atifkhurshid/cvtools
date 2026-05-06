"""
SimCLR Loss implememntation in PyTorch.
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


class SimCLRLoss(nn.Module):

    def __init__(
            self,
            normalize: bool = True,
            temperature: float = 1.0,
            large_number: float = 1e9,
            device: str = "cpu",
        ):
        """
        SimCLR loss function.

        Adapted from: https://github.com/google-research/simclr/blob/master/objective.py
        
        Parameters
        ----------
        normalize : bool, optional
            Whether to L2 normalize the input features, by default True
        temperature : float, optional
            Temperature parameter for scaling the logits, by default 1.0
        large_number : float, optional
            A large number to mask out self-similarities, by default 1e9
        
        Examples
        --------
        >>> criterion = SimCLRLoss(normalize=True, temperature=0.5)
        >>> loss = criterion(features)
        >>> print(loss)
        """
        super().__init__()

        self.normalize = normalize
        self.temperature = temperature
        self.large_number = large_number
        self.device = device

    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute the SimCLR loss.
        
        Parameters
        ----------
        input : torch.Tensor
            Input features of shape (2*batch_size, feature_dim), where the first half
            corresponds to one set of augmented samples and the second half to another set.
        
        Returns
        -------
        torch.Tensor
            The computed SimCLR loss.
        """
        if self.normalize:
            # L2 normalization
            input = F.normalize(input, p=2, dim=1)
        
        batch_size = input.shape[0] // 2
        input1, input2 = torch.split(input, batch_size, dim=0)

        labels = F.one_hot(torch.arange(batch_size), num_classes=batch_size*2).to(self.device).float()
        masks = F.one_hot(torch.arange(batch_size), num_classes=batch_size).to(self.device)

        logits_aa = torch.mm(input1, input1.t()) / self.temperature
        logits_aa = logits_aa - masks * self.large_number
        logits_bb = torch.mm(input2, input2.t()) / self.temperature
        logits_bb = logits_bb - masks * self.large_number
        logits_ab = torch.mm(input1, input2.t()) / self.temperature
        logits_ba = torch.mm(input2, input1.t()) / self.temperature

        loss_a = F.cross_entropy(torch.cat([logits_ab, logits_aa], dim=1), labels)
        loss_b = F.cross_entropy(torch.cat([logits_ba, logits_bb], dim=1), labels)
        loss = loss_a + loss_b

        return loss
    