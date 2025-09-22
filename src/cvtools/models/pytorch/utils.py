"""
Utility functions for PyTorch models.
"""

# Author: Atif Khurshid
# Created: 2025-06-22
# Modified: 2025-09-22
# Version: 1.7
# Changelog:
#     - 2025-08-01: Added type hints and documentation.
#     - 2025-08-01: Updated training loop to include epochs.
#     - 2025-08-08: Added feature map saving functionality.
#     - 2025-08-29: Updated training and evaluation functions.
#     - 2025-08-29: Added training history visualization.
#     - 2025-09-04: Added on_epoch_end function call in training loop.
#     - 2025-09-05: Changed feature map extraction function.
#     - 2025-09-22: Changed validation logic in training function.

from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from .model import PyTorchModel
from ...utils.pytorch import InfiniteDataLoader


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


def evaluate_classification_model(
        model: PyTorchModel,
        data: DataLoader | tuple,
        report: bool = True,
        pbar: bool = True,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate a classification model on the given dataset.

    Requires the model to implement the `eval_step` method.

    Parameters
    -----------
    model : PyTorchModel
        The PyTorch model to evaluate.
    data : DataLoader | tuple 
        DataLoader containing the evaluation data or a tuple (X, y).
    report : bool
        Whether to print the classification report. Default is True.
    pbar : bool
        Whether to show a progress bar during evaluation. Default is True.

    Returns
    --------
    tuple[float, float, np.ndarray, np.ndarray]
        The loss and metric values for the evaluation, along with the predicted and true labels.

    Examples
    ---------
    >>> model = PyTorchSequentialModel([
    ...     nn.Linear(10, 20),
    ...     nn.ReLU(),
    ...     nn.Linear(20, 1)
    ... ])
    >>> evaluate_classification_model(model, dataloader)
    """
    model.eval()
    model.metric.reset()

    if isinstance(data, DataLoader):
        n_batches = len(data)
        y_pred, y_true = [], []
        loss = 0.0
        with torch.no_grad():
            for X, y in tqdm(data, total=n_batches, disable=not pbar):
                batch_pred, batch_loss = model.eval_step(X, y)
                y_pred.extend(batch_pred.numpy())
                y_true.extend(y.numpy())
                loss += batch_loss
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        loss /= n_batches
    else:
        X, y = data
        with torch.no_grad():
            y_pred, loss = model.eval_step(X, y)
        y_pred = y_pred.numpy()
        y_true = y.numpy()

    metric = model.compute_metric()

    if report:
        print(f"Test Loss: {loss:>7f}, Test Metric: {metric:>7f}")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    model.train()

    return loss, metric, y_pred, y_true


def train_classification_model(
        model: PyTorchModel,
        train_dataloader: DataLoader,
        epochs: int = 1,
        val_dataloader: DataLoader | None = None,
        writer: SummaryWriter | None = None,
        log_interval: int = 10,
        iteration_num: int = 0,
    ) -> int:
    """
    Train a PyTorch classification model using the provided dataloader.

    Requires the model to implement the `train_step` and `eval_step` methods.

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
    writer : SummaryWriter | None
        TensorBoard SummaryWriter for logging. If None, no logging is performed.
        Default is None.
    log_interval : int
        Interval (in steps) at which to log training progress. Default is 10.
    iteration_num : int
        Starting iteration number for logging. Default is 0.
    
    Returns
    -------
    int
        The final iteration number after training.

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
        val_dataloader = InfiniteDataLoader(val_dataloader)

    for epoch in range(epochs):
        model.train()

        train_loss = []
        val_loss = []
        train_metric = []
        val_metric = []

        if writer is not None:
            writer.add_scalar('Learning Rate', model.get_learning_rate(), iteration_num)

        for X, y in tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}"):
            batch_loss = model.train_step(X, y)
            batch_metric = model.compute_metric()
            train_loss.append(batch_loss)
            train_metric.append(batch_metric)

            if iteration_num % log_interval == 0:

                if writer is not None:
                    writer.add_scalar('Train Loss', batch_loss, iteration_num)
                    writer.add_scalar('Train Metric', batch_metric, iteration_num)

                if val_dataloader is not None:
                    val_batch = next(val_dataloader)
                    batch_loss, batch_metric, _, _ = evaluate_classification_model(
                        model, val_batch, report=False, pbar=False)
                    val_loss.append(batch_loss)
                    val_metric.append(batch_metric)

                    if writer is not None:
                        writer.add_scalar('Val Loss', batch_loss, iteration_num)
                        writer.add_scalar('Val Metric', batch_metric, iteration_num)

            iteration_num += 1

        if writer is not None:
            writer.close()

        train_loss = np.mean(train_loss)
        train_metric = np.mean(train_metric)

        print(f"Epoch {epoch+1}/{epochs}, ", end="")
        print(f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}, ", end="")
        if val_dataloader is not None:
            val_loss = np.mean(val_loss)
            val_metric = np.mean(val_metric)
            print(f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}", end="")
        print()

        model.on_epoch_end()
    
    model.eval()

    return iteration_num
