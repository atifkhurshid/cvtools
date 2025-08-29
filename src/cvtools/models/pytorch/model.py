"""
Base class for PyTorch models.
"""

# Author: Atif Khurshid
# Created: 2025-06-22
# Modified: 2025-08-29
# Version: 2.0
# Changelog:
#     - 2025-08-29: Added training and evaluation steps.

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Optimizer
from torcheval.metrics.metric import Metric


class PyTorchModel(nn.Module):
    
    def __init__(self):
        """
        Base class for PyTorch models.
        """
        super().__init__()

        self.configured: bool
        self.loss: nn.Module
        self.optimizer: Optimizer
        self.metric: Metric
        self.device: str


    def configure_training(
            self,
            loss: nn.Module,
            optimizer: Optimizer,
            metric: Metric,
            device: str = "cpu",
        ):
        """
        Configure the model with the given loss, optimizer, and metric.

        Parameters
        ----------
        loss : nn.Module
            The loss function to use.
        optimizer : torch.optim.Optimizer
            The optimizer to use.
        metric : callable
            The metric to use for evaluation.
        """
        self.loss = loss
        self.optimizer = optimizer
        self.metric = metric
        self.device = device
        self.configured = True


    def train_step(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """
        Perform a single training step.

        Parameters
        ----------
        X : torch.Tensor
            The input tensor.
        y : torch.Tensor
            The target tensor.

        Returns
        -------
        float
            The loss value.
        """
        if not self.configured:
            raise RuntimeError("Model is not configured for training.")

        self.optimizer.zero_grad()

        X = X.to(self.device)
        y = y.to(self.device)

        outputs = self.forward(X)

        loss = self.compute_loss(outputs, y)
        loss.backward()

        self.optimizer.step()

        self.metric.update(outputs, y)

        return loss.item()
    

    def eval_step(self, X: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Evaluate the model on the given input and target tensors.

        Parameters
        ----------
        X : torch.Tensor
            The input tensor.
        y : torch.Tensor
            The target tensor.

        Returns
        -------
        tuple[torch.Tensor, float]
            The predicted labels and the loss value.
        """
        if not self.configured:
            raise RuntimeError("Model is not configured for evaluation.")

        X = X.to(self.device)
        y = y.to(self.device)

        outputs = self.forward(X)

        loss = self.compute_loss(outputs, y)

        _, preds = torch.max(outputs, 1)
        preds = preds.detach().cpu()

        self.metric.update(outputs, y)

        return preds, loss.item()


    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss for the given outputs and targets.

        Parameters
        ----------
        outputs : torch.Tensor
            The model outputs.
        targets : torch.Tensor
            The ground truth targets.

        Returns
        -------
        torch.Tensor
            The computed loss.
        """
        return self.loss(outputs, targets)
    

    def compute_metric(self) -> float:
        """
        Compute and reset the metric.

        Returns
        -------
        float
            The computed metric value.
        """
        metric = self.metric.compute()
        self.metric.reset()

        return metric.item()
    