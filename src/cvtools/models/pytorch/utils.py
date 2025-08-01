"""
Utility functions for PyTorch models.
"""

# Author: Atif Khurshid
# Created: 2025-06-22
# Modified: 2025-08-01
# Version: 1.2
# Changelog:
#     - 2025-08-01: Added type hints and documentation.
#     - 2025-08-01: Updated training loop to include epochs.

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from .model import PyTorchModel


def extract_features(
        model: nn.Module,
        dataloader: DataLoader,
        device: str,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract output features from the model for each sample in the dataloader.

    Parameters:
    -----------
    model : nn.Module
        The PyTorch model to extract features from.
    dataloader : DataLoader
        DataLoader containing the dataset.
    device : str
        Device to run the model on (e.g., 'cpu' or 'cuda').

    Returns:
    --------
    tuple
        A tuple containing:
        - features: numpy array of shape (n_samples, n_features)
        - labels: numpy array of shape (n_samples,)

    Examples:
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


def evaluate_classification_model(
        model: PyTorchModel,
        dataloader: DataLoader,
        device: str,
        report: bool = True,
    ) -> float | None:
    """
    Evaluate a classification model on the given dataloader.

    Requires the model to implement the `eval_step` method.

    Parameters:
    -----------
    model : PyTorchModel
        The PyTorch model to evaluate.
    dataloader : DataLoader
        DataLoader containing the dataset.
    device : str
        Device to run the model on (e.g., 'cpu' or 'cuda').
    report : bool
        Whether to print the classification report.

    Returns:
    --------
    float | None
        The average loss over the dataset if report is False, otherwise None.
    
    Examples:
    ---------
    >>> model = PyTorchSequentialModel([
    ...     nn.Linear(10, 20),
    ...     nn.ReLU(),
    ...     nn.Linear(20, 1)
    ... ])
    >>> dataloader = DataLoader(dataset, batch_size=32)
    >>> loss = evaluate_classification_model(model, dataloader, device='cuda', report=True)
    >>> print(f"Average Loss: {loss:.4f}")
    """
    model = model.to(device)
    model.eval()

    n_batches = len(dataloader)
    y_pred, y_true = [], []
    loss = 0
    with torch.no_grad():
        for X, y in tqdm(dataloader, total=n_batches):
            X = X.to(device)
            y = y.to(device)
            batch_pred, batch_loss = model.eval_step(X, y)
            y_pred.extend(batch_pred)
            y_true.extend(y)
            loss += batch_loss

    loss /= n_batches

    if report:
        print(f"Test Loss: {loss:>7f}")
        print(classification_report(y_true, y_pred))
    else:
        return loss


def train_model(
        model: PyTorchModel,
        train_dataloader: DataLoader,
        device: str,
        epochs: int = 1,
        val_dataloader: DataLoader | None = None,
    ):
    """
    Train a PyTorch classification model using the provided dataloader.

    Requires the model to implement the `train_step` and `eval_step` methods.

    Parameters:
    -----------
    model : PyTorchModel
        The PyTorch model to train.
    train_dataloader : DataLoader
        DataLoader containing the training dataset.
    device : str
        Device to run the model on (e.g., 'cpu' or 'cuda').
    epochs : int
        Number of epochs to train the model.
    val_dataloader : DataLoader | None
        DataLoader containing the validation dataset. If None, no validation is performed.
    
    Examples:
    ---------
    >>> model = PyTorchSequentialModel([
    ...     nn.Linear(10, 20),
    ...     nn.ReLU(),
    ...     nn.Linear(20, 1)
    ... ])
    >>> train_dataloader = DataLoader(train_dataset, batch_size=32)
    >>> val_dataloader = DataLoader(val_dataset, batch_size=32)
    >>> train_model(model, train_dataloader, device='cuda', epochs=10, val_dataloader=val_dataloader)
    >>> print("Training complete.")
    """
    model = model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0.0

        for X, y in tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch+1}/{epochs}"):
            X = X.to(device)
            y = y.to(device)

            batch_loss = model.train_step(X, y)
            train_loss += batch_loss.item()

        if val_dataloader is not None:
            val_loss = evaluate_classification_model(model, val_dataloader, device, report=False)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_dataloader)}, Val Loss: {val_loss}")
