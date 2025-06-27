"""
Driver functions for PyTorch models.
"""

# Author: Atif Khurshid
# Created: 2025-06-22
# Modified: None
# Version: 1.0
# Changelog:
#     - None

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from .model import PyTorchModel


def extract_features(
        model: nn.Module,
        dataloader: DataLoader,
        device: str,
    ):

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


def evaluate_model(
        model: PyTorchModel,
        dataloader: DataLoader,
        device: str,
        report: bool = True,
    ):

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
        val_dataloader: DataLoader | None = None,
        val_step: int = 100,
    ):

    model = model.to(device)
    model.train()

    for batch, (X, y) in enumerate(train_dataloader):
        X = X.to(device)
        y = y.to(device)
        train_loss = model.train_step(X, y)

        if val_dataloader is not None and batch % val_step == 0:
            val_loss = evaluate_model(model, val_dataloader, device, report=False)

            print("[{:d}]/[{:d}] Train Loss: {:.3f} - Test Loss: {:.3f}".format(
                batch + 1, len(train_dataloader), train_loss, val_loss
            ))
    