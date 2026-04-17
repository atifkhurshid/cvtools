"""
Base class for PyTorch models.
"""

# Author: Atif Khurshid
# Created: 2025-06-22
# Modified: 2026-04-17
# Version: 2.3
# Changelog:
#     - 2025-08-29: Added training and evaluation steps.
#     - 2025-09-04: Added scheduler support.
#     - 2026-03-05: Added support for returning logits for evaluation.
#     - 2026-04-16: Added support for non-blocking transfers to device.
#     - 2026-04-17: Added support for on-GPU preprocessing.

from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torcheval.metrics.metric import Metric


class PyTorchModel(nn.Module):
    
    def __init__(self):
        """
        Base class for PyTorch models.
        """
        super().__init__()

        self.loss: nn.Module
        self.optimizer: Optimizer
        self.metric: Metric
        self.scheduler: Optional[LRScheduler]
        self.device: str
        self.non_blocking: bool
        self.preprocess_train: Optional[nn.Module]
        self.preprocess_test: Optional[nn.Module]

        self.configured: bool = False
        self.training_samples_seen: int = 0


    def configure_training(
            self,
            loss: nn.Module,
            optimizer: Optimizer,
            metric: Metric,
            scheduler: Optional[LRScheduler] = None,
            device: str = "cpu",
            non_blocking: bool = False,
            preprocess_train: Optional[nn.Module] = None,
            preprocess_test: Optional[nn.Module] = None,
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
        scheduler : torch.optim.lr_scheduler.LRScheduler, optional
            The learning rate scheduler to use, by default None.
        device : str, optional
            The device to use, by default "cpu".
        non_blocking : bool, optional
            Whether to use non-blocking transfers to the device, by default False.
        preprocess_train : nn.Module, optional
            The preprocessing module to use for training data, by default None.
        preprocess_test : nn.Module, optional
            The preprocessing module to use for test data, by default None.
        """
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metric = metric
        self.device = device
        self.non_blocking = non_blocking
        self.preprocess_train = preprocess_train
        self.preprocess_test = preprocess_test

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
        self.optimizer.zero_grad()

        X, y = self.prepare_step(X, y)

        if self.preprocess_train is not None:
            X = self.preprocess_train(X)

        outputs = self.forward(X)

        loss = self.compute_loss(outputs, y)
        loss.backward()

        self.optimizer.step()

        self.metric.update(outputs, y)

        self.training_samples_seen += len(X)

        return loss.item()
    

    def test_step(
            self,
            X: torch.Tensor, 
            y: torch.Tensor,
            return_outputs: bool = False,
        ) -> tuple[Optional[torch.Tensor], float, torch.Tensor]:
        """
        Perform a single test step.

        Parameters
        ----------
        X : torch.Tensor
            The input tensor.
        y : torch.Tensor
            The target tensor.
        return_outputs : bool, optional
            Whether to return the model outputs, by default False.

        Returns
        -------
        tuple[Optional[torch.Tensor], float, torch.Tensor]
            The model outputs (None if return_outputs is False), the loss value, and the predicted labels.
        """
        X, y = self.prepare_step(X, y)

        if self.preprocess_test is not None:
            X = self.preprocess_test(X)

        outputs = self.forward(X)

        loss = self.compute_loss(outputs, y)

        _, preds = torch.max(outputs, 1)
        preds = preds.detach().cpu()

        self.metric.update(outputs, y)

        if return_outputs:
            outputs = outputs.detach().cpu()
        else:
            outputs = None
        
        return outputs, loss.item(), preds


    def prepare_step(
            self,
            X: torch.Tensor,
            y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare the input and target tensors for a training or test step.

        Parameters
        ----------
        X : torch.Tensor
            The input tensor.
        y : torch.Tensor
            The target tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The prepared input and target tensors.
        """
        if not self.configured:
            raise RuntimeError("Model is not configured for training.")

        X = X.to(self.device, non_blocking=self.non_blocking)
        y = y.to(self.device, non_blocking=self.non_blocking)

        return X, y


    def compute_loss(
            self,
            outputs: torch.Tensor,
            targets: torch.Tensor
        ) -> torch.Tensor:
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
    

    def on_epoch_end(self):
        """
        Function to be called at the end of each epoch.
        """
        if self.scheduler is not None:
            self.scheduler.step()


    def get_learning_rate(self) -> float:
        """
        Get the current learning rate.

        Returns
        -------
        float
            The current learning rate.
        """
        return self.optimizer.param_groups[0]['lr']
