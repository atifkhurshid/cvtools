"""
Utility functions for PyTorch models.
"""

# Author: Atif Khurshid
# Created: 2025-06-22
# Modified: 2025-09-05
# Version: 1.6
# Changelog:
#     - 2025-08-01: Added type hints and documentation.
#     - 2025-08-01: Updated training loop to include epochs.
#     - 2025-08-08: Added feature map saving functionality.
#     - 2025-08-29: Updated training and evaluation functions.
#     - 2025-08-29: Added training history visualization.
#     - 2025-09-04: Added on_epoch_end function call in training loop.
#     - 2025-09-05: Changed feature map extraction function.

from pathlib import Path

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, Subset

from .model import PyTorchModel


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
        dataset: Dataset,
        batch_size: int,
        num_batches: int | None = None,
        shuffle: bool = False,
        report: bool = True,
        pbar: bool = True,
        **kwargs: dict,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate a classification model on the given dataset.

    Requires the model to implement the `eval_step` method.

    Parameters
    -----------
    model : PyTorchModel
        The PyTorch model to evaluate.
    dataset : Dataset
        Dataset containing the data to evaluate.
    batch_size : int
        Batch size to use for evaluation.
    num_batches : int | None
        Number of batches to use for evaluation. If None, use all batches.
        Default is None.
    shuffle : bool
        Whether to shuffle the data before evaluation. Default is False.
    report : bool
        Whether to print the classification report. Default is True.
    pbar : bool
        Whether to show a progress bar during evaluation. Default is True.
    **kwargs : dict
        Additional keyword arguments passed to the DataLoader.

    Returns
    --------
    tuple[float, float]
        The loss and metric values for the evaluation.

    Examples
    ---------
    >>> model = PyTorchSequentialModel([
    ...     nn.Linear(10, 20),
    ...     nn.ReLU(),
    ...     nn.Linear(20, 1)
    ... ])
    >>> evaluate_classification_model(model, dataset, batch_size=32)
    """
    model.eval()

    if num_batches is not None:
        indices = np.random.choice(len(dataset), num_batches * batch_size, replace=False)
        dataset = Subset(dataset, indices)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    n_batches = len(dataloader)
    y_pred, y_true = [], []
    loss = 0.0
    with torch.no_grad():
        for X, y in tqdm(dataloader, total=n_batches, disable=not pbar):
            batch_pred, batch_loss = model.eval_step(X, y)
            y_pred.extend(batch_pred.numpy())
            y_true.extend(y.numpy())
            loss += batch_loss

    loss /= n_batches
    metric = model.compute_metric()

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    if report:
        print(f"Test Loss: {loss:>7f}, Test Metric: {metric:>7f}")
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    return loss, metric, y_pred, y_true


def train_classification_model(
        model: PyTorchModel,
        train_dataset: Dataset,
        epochs: int = 1,
        batch_size: int = 32,
        shuffle_train: bool = True,
        val_dataset: Dataset | None = None,
        val_batches: int | None = 10,
        shuffle_val: bool = False,
        writer: SummaryWriter | None = None,
        log_interval: int = 10,
        iteration_num: int = 0,
        **kwargs: dict,
    ) -> int:
    """
    Train a PyTorch classification model using the provided dataloader.

    Requires the model to implement the `train_step` and `eval_step` methods.

    Parameters:
    -----------
    model : PyTorchModel
        The PyTorch model to train.
    train_dataset : Dataset
        Dataset containing the training data.
    epochs : int
        Number of epochs to train the model. Default is 1.
    batch_size : int
        Batch size to use for training. Default is 32.
    shuffle_train : bool
        Whether to shuffle the training data. Default is True.
    val_dataset : Dataset | None
        Dataset containing the validation data. If None, no validation is performed.
        Default is None.
    val_batches : int | None
        Number of batches to use for validation. Default is 10.
    shuffle_val : bool
        Whether to shuffle the validation data. Default is False.
    writer : SummaryWriter | None
        TensorBoard SummaryWriter for logging. If None, no logging is performed.
        Default is None.
    log_interval : int
        Interval (in steps) at which to log training progress. Default is 10.
    iteration_num : int
        Starting iteration number for logging. Default is 0.
    **kwargs : dict
        Additional keyword arguments passed to the DataLoader.
    
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
    >>> train_classification_model(model, train_dataset, val_dataset=val_dataset)
    """
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, **kwargs)

    for epoch in range(epochs):
        model.train()

        train_loss = 0.0
        train_metric = 0.0

        if writer is not None:
            writer.add_scalar('Learning Rate', model.get_learning_rate(), iteration_num)

        for X, y in tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}"):
            batch_loss = model.train_step(X, y)
            batch_metric = model.compute_metric()

            train_loss += batch_loss
            train_metric += batch_metric

            if writer is not None and iteration_num % log_interval == 0:
                writer.add_scalar('Train Loss', batch_loss, iteration_num)
                writer.add_scalar('Train Metric', batch_metric, iteration_num)

            iteration_num += 1

        if val_dataset is not None:
            val_loss, val_metric, _, _ = evaluate_classification_model(
                model, val_dataset, batch_size, val_batches, shuffle_val, report=False, pbar=False, **kwargs)
            
            if writer is not None:
                writer.add_scalar('Val Loss', val_loss, iteration_num)
                writer.add_scalar('Val Metric', val_metric, iteration_num)

        if writer is not None:
            writer.close()

        train_loss /= len(train_dataloader)
        train_metric /= len(train_dataloader)

        print(f"Epoch {epoch+1}/{epochs}, ", end="")
        print(f"Train Loss: {train_loss:.4f}, Train Metric: {train_metric:.4f}, ", end="")
        if val_dataset is not None:
            print(f"Val Loss: {val_loss:.4f}, Val Metric: {val_metric:.4f}", end="")
        print()

        model.on_epoch_end()
    
    return iteration_num
