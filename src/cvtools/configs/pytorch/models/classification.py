"""
Model Configuration Class for PyTorch Classification Models.
"""

# Author: Atif Khurshid
# Created: 2026-05-12
# Modified: 2026-10-22
# Version: 1.3
# Changelog:
#     - 2026-05-12: Initial version.
#     - 2026-07-30: Added support for binary cross-entropy loss function.
#     - 2026-09-25: Added support for more scheduler types.
#     - 2026-10-09: Added support for pretraining.
#     - 2026-10-22: Added documentation for all parameters in the constructor.

from typing import Optional

import torch
import torch.nn as nn
import torcheval.metrics as metrics

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import Optimizer
from torch.optim import SGD
from torch.optim import AdamW

from cvtools.models.pytorch import PyTorchModel

from ..._base import ConfigBase


class PyTorchClassificationModelConfig(ConfigBase):

    def __init__(self, **kwargs):
        """
        This class is meant to be a single-call interface that defines the configuration for
        a PyTorch classification model, including model architecture, loss function, optimizer,
        metric, and scheduler.

        Simply set the configuration parameters and call the `prepare_model()` method to get a
        PyTorch model ready for training.

        Parameters
        ----------
        model_name : str
            The name of the model architecture to use.
        model_class : type[PyTorchModel]
            The class of the model architecture to use.
        model_kwargs : dict
            The keyword arguments to pass to the model class constructor.
        n_classes : int
            The number of classes in the classification task.
        input_shape : tuple
            The shape of the input data (batch_size, channels, height, width).
        device : torch.device
            The device on which to train the model (e.g., "cpu" or "cuda").
        loss_fn : str
            The loss function to use for training. Options: "cross_entropy", "binary_cross_entropy".
        class_weights : Optional[torch.Tensor]
            The class weights to use for the loss function (if applicable).
        optimizer : str
            The optimizer to use for training. Options: "sgd", "adamw".
        metric : str
            The metric to use for evaluation. Options: "multiclass_accuracy", "multilabel_accuracy", "binary_accuracy".
        epochs : int
            The number of epochs to train the model.
        learning_rate : float
            The learning rate for the optimizer.
        momentum : float
            The momentum for the SGD optimizer (if used).
        weight_decay : float
            The weight decay (L2 penalty) for the optimizer.
        multilabel : bool
            Whether the classification task is multilabel (True) or multiclass (False).
        non_blocking : bool
            Whether to use non-blocking transfers for data loading.
        val_strategy : str
            The validation strategy to use. Options: "dataset", "split".
        log_frequency : int
            The frequency (in batches) at which to log training progress.
        scheduler : Optional[str] 
            The learning rate scheduler to use. Options: "step", "multistep", "plateau", or None.
        scheduler_step_size : Optional[int]
            The step size for the StepLR scheduler (if used).
        scheduler_milestones : Optional[list]
            The milestones for the MultiStepLR scheduler (if used).
        scheduler_gamma : float
            The gamma value for the learning rate scheduler.
        scheduler_mode : str
            The mode for the ReduceLROnPlateau scheduler. Options: "min", "max".
        scheduler_factor : float
            The factor by which to reduce the learning rate for the ReduceLROnPlateau scheduler.
        scheduler_patience : int
            The number of epochs with no improvement after which learning rate will be reduced for the ReduceLROnPlateau scheduler.
        scheduler_threshold : float
            The threshold for measuring the new optimum for the ReduceLROnPlateau scheduler.
        scheduler_threshold_mode : str
            The mode for the threshold for the ReduceLROnPlateau scheduler. Options: "rel", "abs".
        scheduler_cooldown : int
            The number of epochs to wait before resuming normal operation after the learning rate has been reduced for the ReduceLROnPlateau scheduler.
        scheduler_min_lr : float
            The minimum learning rate for the ReduceLROnPlateau scheduler.
        scheduler_eps : float
            The epsilon value for the ReduceLROnPlateau scheduler.
        pretrain : bool
            Whether to use pretraining for the model.
        pretrain_epochs : int
            The number of epochs to pretrain the model (if pretrain is True).
        early_stopping : bool
            Whether to use early stopping during training.
        patience : int
            The number of epochs with no improvement after which training will be stopped for early stopping.
        min_delta : float
            The minimum change in the monitored metric to qualify as an improvement for early stopping.
        restore_best_weights : bool
            Whether to restore the model weights from the epoch with the best value of the monitored metric for early stopping.
        checkpoint_dir : Optional[str]
            The directory where model checkpoints will be saved during training.
        checkpoint_interval : int
            The interval (in epochs) at which to save model checkpoints during training.
        """
        super().__init__()

        self.model_name: str = ""
        self.model_class: Optional[type[PyTorchModel]] = None
        self.model_kwargs: dict = {}
        self.n_classes: int = 0
        self.input_shape: tuple = (1, 3, 224, 224)
        self.device: torch.device = torch.device("cpu")

        self.loss_fn: str = "cross_entropy"
        self.class_weights: Optional[torch.Tensor] = None
        self.optimizer: str = "sgd"
        self.metric: str = "multiclass_accuracy"
        self.epochs: int = 0
        self.learning_rate: float = 1e-2
        self.momentum: float = 0.9
        self.weight_decay: float = 1e-4
        self.multilabel: bool = False

        self.non_blocking: bool = True
        self.val_strategy: str = "dataset"
        self.log_frequency: int = 5

        self.scheduler: Optional[str] = None
        self.scheduler_step_size: Optional[int] = 10
        self.scheduler_milestones: Optional[list] = None
        self.scheduler_gamma: float = 0.1
        self.scheduler_mode: str = "min"
        self.scheduler_factor: float = 0.1
        self.scheduler_patience: int = 10
        self.scheduler_threshold: float = 1e-4
        self.scheduler_threshold_mode: str = "rel"
        self.scheduler_cooldown: int = 0
        self.scheduler_min_lr: float = 0.0
        self.scheduler_eps: float = 1e-8

        self.pretrain: bool = False
        self.pretrain_epochs: int = 0

        self.early_stopping: bool = False
        self.early_stopping_patience: int = 10
        self.early_stopping_min_delta: float = 1e-4
        self.early_stopping_restore_best_weights: bool = True

        self.checkpoint_dir: Optional[str] = None
        self.checkpoint_interval: int = 10

        if type(self) is PyTorchClassificationModelConfig:
            # Only run if this is the instantiated class,
            # and not a parent class being inherited from.
            self._update_from_kwargs(**kwargs)


    def prepare_model(self) -> PyTorchModel:
        """
        Prepares the model for training by defining the model architecture,
        loss function, optimizer, metric, and scheduler based on the configuration.

        Returns
        -------
        PyTorchModel
            The prepared model ready for training.
        """
        model = self._define_model()
        model = model.to(self.device)

        loss_fn = self._define_loss_fn(self.loss_fn)
        optimizer = self._define_optimizer(model, self.optimizer)
        metric = self._define_metric(self.metric)
        scheduler = self._define_scheduler(optimizer)

        model.configure_training(
            loss = loss_fn,
            optimizer = optimizer,
            metric = metric,
            multilabel = self.multilabel,
            scheduler = scheduler,
            device = self.device,
            non_blocking = self.non_blocking,
        )

        return model


    def _define_model(self) -> PyTorchModel:
        """
        Defines the model architecture based on the configuration.

        Returns
        -------
        PyTorchModel
            The defined model.
        """
        if self.model_class is None:
            raise ValueError("Model class is not defined in the configuration.")
        return self.model_class(**self.model_kwargs)


    def _define_loss_fn(self, loss_fn_name: str):
        """
        Defines the loss function based on the configuration.

        Parameters
        ----------
        loss_fn_name : str
            The name of the loss function to define.

        Returns
        -------
        nn.Module
            The defined loss function.
        """
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.device)
            
        if loss_fn_name == "cross_entropy":
            return nn.CrossEntropyLoss(weight = self.class_weights)
        elif loss_fn_name == "binary_cross_entropy":
            return nn.BCEWithLogitsLoss(pos_weight = self.class_weights)
        else:
            raise ValueError(f"Unsupported loss function(s): {loss_fn_name}")


    def _define_metric(self, metric_name: str):
        """
        Defines the metric based on the configuration.

        Parameters
        ----------
        metric_name : str
            The name of the metric to define.
        
        Returns
        -------
        Metric
            The defined metric.
        """
        if metric_name == "multilabel_accuracy":
            return metrics.MultilabelAccuracy(
                threshold = 0.5,
                criteria = "exact_match",
            )
        elif metric_name == "multiclass_accuracy":
            return metrics.MulticlassAccuracy(
                average = "macro",
                num_classes = self.n_classes,
            )
        elif metric_name == "binary_accuracy":
            return metrics.BinaryAccuracy(
                threshold = 0.5,
            )
        else:
            raise ValueError(f"Unsupported metric(s): {metric_name}")


    def _define_optimizer(self, model: PyTorchModel, optimizer_name: str):
        """
        Defines the optimizer based on the configuration.

        Parameters
        ----------
        model : PyTorchModel
            The model for which to define the optimizer.
        optimizer_name : str
            The name of the optimizer to define.

        Returns
        -------
        Optimizer
            The defined optimizer.
        """
        if optimizer_name == "sgd":

            return SGD(
                model.parameters(),
                lr = self.learning_rate,
                momentum = self.momentum,
                weight_decay = self.weight_decay
            )
        
        elif optimizer_name == "adamw":

            return AdamW(
                model.parameters(),
                lr = self.learning_rate,
                weight_decay = self.weight_decay
            )

        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")


    def _define_scheduler(self, optimizer: Optimizer):
        """
        Defines the learning rate scheduler based on the configuration.

        Parameters
        ----------
        optimizer : Optimizer
            The optimizer for which to define the scheduler.
        """
        if self.scheduler == "step":
            if self.scheduler_step_size is None:
                raise ValueError("scheduler_step_size must be specified for StepLR scheduler.")
            scheduler = StepLR(
                optimizer,
                step_size = self.scheduler_step_size,
                gamma = self.scheduler_gamma
            )

        elif self.scheduler == "multistep":
            if self.scheduler_milestones is None:
                raise ValueError("scheduler_milestones must be specified for MultiStepLR scheduler.")
            scheduler = MultiStepLR(
                optimizer,
                milestones = self.scheduler_milestones,
                gamma = self.scheduler_gamma
            )

        elif self.scheduler == "plateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode = self.scheduler_mode,
                factor = self.scheduler_factor,
                patience = self.scheduler_patience,
                threshold = self.scheduler_threshold,
                threshold_mode = self.scheduler_threshold_mode,
                cooldown = self.scheduler_cooldown,
                min_lr = self.scheduler_min_lr,
                eps = self.scheduler_eps
            )
        else:
            scheduler = None

        return scheduler
