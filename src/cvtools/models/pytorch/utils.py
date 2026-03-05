"""
Utility functions for PyTorch models.
"""

# Author: Atif Khurshid
# Created: 2025-06-22
# Modified: 2026-03-05
# Version: 2.1
# Changelog:
#     - 2025-08-01: Added type hints and documentation.
#     - 2025-08-01: Updated training loop to include epochs.
#     - 2025-08-08: Added feature map saving functionality.
#     - 2025-08-29: Updated training and evaluation functions.
#     - 2025-08-29: Added training history visualization.
#     - 2025-09-04: Added on_epoch_end function call in training loop.
#     - 2025-09-05: Changed feature map extraction function.
#     - 2025-09-22: Changed validation logic in training function.
#     - 2025-10-15: Replaced tensorboard with wandb for logging.
#     - 2026-03-02: Allowed validation on entire dataset.
#     - 2026-03-04: Added early stopping functionality to training loop.
#     - 2026-03-05: Updated test_classification_model to optionally return logits for evaluation.

from pathlib import Path
from typing import Union, Optional

import wandb
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from .base import PyTorchModel
from ...utils.pytorch import InfiniteDataLoader
from ...utils.pytorch import EarlyStopping


def extract_features(
        model: nn.Module,
        dataloader: DataLoader,
        device: str,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract output features from the model for each sample in the dataloader.

    Parameters
    -----------
    model : nn.Module
        The PyTorch model to extract features from.
    dataloader : DataLoader
        DataLoader containing the dataset.
    device : str
        Device to run the model on (e.g., 'cpu' or 'cuda').

    Returns
    --------
    tuple
        A tuple containing:
        - features: numpy array of shape (n_samples, n_features)
        - labels: numpy array of shape (n_samples,)

    Examples
    ---------
    >>> model = PyTorchSequentialModel([
    ...     nn.Linear(10, 20),
    ...     nn.ReLU(),
    ...     nn.Linear(20, 1)
    ... ])
    >>> dataloader = DataLoader(dataset, batch_size=32)
    >>> features, labels = extract_features(model, dataloader, device='cuda')
    >>> print(features.shape, labels.shape)
    (n_samples, n_features) (n_samples,)
    """
    model = model.to(device)
    model.eval()

    features = []
    labels = []
    for X, y in tqdm(dataloader, total=len(dataloader)):
        X = X.to(device)
        with torch.no_grad():
            f = model(X)
        features.append(f.cpu().numpy())
        labels.append(y.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels


def save_feature_maps(
        model: nn.Module,
        dataset: Dataset,
        save_dir: str,
        batch_size: int = 32,
        device: str = 'cpu',
    ) -> None:
    """
    Save the output feature maps from the model for each image in the dataset.

    Feature maps are saved as separate `.npy` files for each image.

    Parameters
    -----------
    model : nn.Module
        The PyTorch model to extract feature maps from.
    dataset : Dataset
        Dataset containing the images.
    save_path : str
        Path where the feature maps will be saved.
    batch_size : int
        Batch size to use for processing the dataset. Default is 32.
    device : str
        Device to run the model on (e.g., 'cpu' or 'cuda').

    Examples
    ---------
    >>> model = PyTorchSequentialModel([
    ...     nn.Conv2d(10, 20, kernel_size=3),
    ...     nn.ReLU(),
    ... ])
    >>> save_feature_maps(model, dataset, save_path='features', device='cuda')
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model = model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    index = 0
    for X, _ in tqdm(dataloader, total=len(dataloader)):
        X = X.to(device)
        with torch.no_grad():
            features = model(X)

        features = features.cpu().numpy()

        for i in range(features.shape[0]):
            np.save(save_dir / f"features_{index}.npy", features[i])
            index += 1


def test_classification_model(
        model: PyTorchModel,
        data: Union[DataLoader, tuple],
        return_outputs: bool = False,
        pbar: bool = True,
    ) -> tuple[float, float, Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Perform inference on a given classification model and dataset.

    Requires the model to implement the `test_step` method.

    Parameters
    -----------
    model : PyTorchModel
        The PyTorch model to test.
    data : DataLoader | tuple 
        DataLoader containing the evaluation data or a tuple (X, y).
    return_outputs : bool
        Whether to return the model outputs along with the predictions and loss. Default is False.
    pbar : bool
        Whether to show a progress bar during evaluation. Default is True.

    Returns
    --------
    tuple[float, float, Optional[np.ndarray], np.ndarray, np.ndarray]
    A tuple containing:
        - loss: The average loss over the dataset.
        - metric: The computed metric value over the dataset.
        - outputs: The model outputs (logits) for each sample if return_outputs is True, otherwise None.
        - y_pred: The predicted labels for each sample.
        - y_true: The true labels for each sample.

    Examples
    ---------
    >>> model = PyTorchSequentialModel([
    ...     nn.Linear(10, 20),
    ...     nn.ReLU(),
    ...     nn.Linear(20, 1)
    ... ])
    >>> test_classification_model(model, dataloader)
    """
    model.eval()
    model.metric.reset()

    if isinstance(data, DataLoader):
        n_batches = len(data)
        outputs = []
        y_pred, y_true = [], []
        loss = 0.0
        with torch.no_grad():
            for X, y in tqdm(data, total=n_batches, disable=not pbar):
                batch_outputs, batch_loss, batch_pred = model.test_step(X, y, return_outputs)
                y_pred.extend(batch_pred.numpy())
                y_true.extend(y.numpy())
                loss += batch_loss
                if return_outputs:
                    outputs.extend(batch_outputs.numpy())
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        loss /= n_batches
        if return_outputs:
            outputs = np.array(outputs)
    else:
        X, y = data
        with torch.no_grad():
            outputs, loss, y_pred = model.test_step(X, y, return_outputs)
        y_pred = y_pred.numpy()
        y_true = y.numpy()
        if return_outputs:
            outputs = outputs.numpy()

    metric = model.compute_metric()

    model.train()

    if not return_outputs:
        outputs = None

    return loss, metric, outputs, y_pred, y_true


def train_classification_model(
        model: PyTorchModel,
        train_dataloader: DataLoader,
        epochs: int = 1,
        val_dataloader: Optional[DataLoader] = None,
        val_strategy: str = "batch",
        early_stopping: bool = False,
        patience: int = 5,
        min_delta: float = 1e-4,
        restore_best_weights: bool = True,
        run: Optional[wandb.Run] = None,
        log_interval: int = 10,
    ):
    """
    Train a PyTorch classification model using the provided dataloader.

    Requires the model to implement the `train_step` and `test_step` methods.

    Parameters:
    -----------
    model : PyTorchModel
        The PyTorch model to train.
    train_dataloader : DataLoader
        DataLoader containing the training data.
    epochs : int
        Number of epochs to train the model. Default is 1.
    val_dataloader : DataLoader | None
        DataLoader containing the validation data. If None, no validation is performed.
        Default is None.
    val_strategy : str
        Strategy for validation:
        - "dataset": Evaluate on the entire validation dataset at each logging interval.
        - "batch": Evaluate on a single batch from the validation dataset at each logging interval.
        Default is "batch".
    run : wandb.Run | None
        Weights & Biases run for logging. If None, no logging is performed.
        Default is None.
    log_interval : int
        Interval (in steps) at which to log training progress. Default is 10.

    Examples
    ---------
    >>> model = PyTorchSequentialModel([
    ...     nn.Linear(10, 20),
    ...     nn.ReLU(),
    ...     nn.Linear(20, 1)
    ... ])
    >>> train_classification_model(model, train_dataloader, val_dataloader=val_dataloader)
    """
    if val_dataloader is not None:
        if val_strategy == "batch":
            val_dataloader = InfiniteDataLoader(val_dataloader)

        if early_stopping:
            early_stopper = EarlyStopping(
                patience = patience,
                min_delta = min_delta,
                restore_best_weights = restore_best_weights
            )

    epoch = 0
    while epoch < epochs:
        epoch += 1
        model.train()

        epoch_loss_train = []
        epoch_loss_val = []
        epoch_metric_train = []
        epoch_metric_val = []

        for i, (X, y) in tqdm(enumerate(train_dataloader),
                              total=len(train_dataloader), desc=f"Epoch {epoch}/{epochs}"):
            batch_loss_train = model.train_step(X, y)
            batch_metric_train = model.compute_metric()
            epoch_loss_train.append(batch_loss_train)
            epoch_metric_train.append(batch_metric_train)

            if i % log_interval == 0:

                if val_dataloader is not None:
                    if val_strategy == "dataset":
                        batch_loss_val, batch_metric_val, _, _, _ = test_classification_model(
                            model, val_dataloader, return_outputs=False, pbar=False)
                        epoch_loss_val = batch_loss_val
                        epoch_metric_val = batch_metric_val
                    elif val_strategy == "batch":
                        X, y = next(val_dataloader)
                        batch_loss_val, batch_metric_val, _, _, _ = test_classification_model(
                            model, (X, y), return_outputs=False, pbar=False)
                        epoch_loss_val.append(batch_loss_val)
                        epoch_metric_val.append(batch_metric_val)

                if run is not None:
                    run.log({
                        "train/loss": batch_loss_train,
                        "train/metric": batch_metric_train,
                        "valid/loss": batch_loss_val if val_dataloader is not None else None,
                        "valid/metric": batch_metric_val if val_dataloader is not None else None,
                        "learning_rate": model.get_learning_rate(),
                        "samples_seen": model.training_samples_seen,
                    })

        epoch_loss_train = np.mean(epoch_loss_train)
        epoch_metric_train = np.mean(epoch_metric_train)

        print(f"Epoch {epoch}/{epochs}, ", end="")
        print(f"Train Loss: {epoch_loss_train:.4f}, Train Metric: {epoch_metric_train:.4f}, ", end="")
        if val_dataloader is not None:
            epoch_loss_val = np.mean(epoch_loss_val)
            epoch_metric_val = np.mean(epoch_metric_val)
            print(f"Val Loss: {epoch_loss_val:.4f}, Val Metric: {epoch_metric_val:.4f}", end="")

            if early_stopping:
                early_stopper.step(model, epoch_loss_val)
                if early_stopper.early_stop:
                    print(f"\nEarly stopping triggered at epoch {epoch}")
                    early_stopper.restore(model)
                    epoch = epochs  # Exit outer loop
        print()

        model.on_epoch_end()
    
    if early_stopping and not early_stopper.early_stop:
        early_stopper.restore(model)

    model.eval()
