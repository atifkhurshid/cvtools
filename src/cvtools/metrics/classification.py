"""
Metrics for evaluating classification models.
"""

# Author: Atif Khurshid
# Created: 2026-03-05
# Modified: 2026-05-05
# Version: 1.2
# Changelog:
#     - 2026-03-05: Added ROC curve and AUC score computation.
#     - 2026-03-12: Added n_interp_points parameter to compute_roc function.
#     - 2026-03-12: Added interpolation of roc curve for micro weighted average.
#     - 2026-05-05: Added multilabel metrics.

from typing import Optional

import torch
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score, auc, accuracy_score


def compute_roc(
        labels: list,
        outputs: np.ndarray,
        mode: str = "binary",
        classes: Optional[list] = None,
        n_interp_points: int = 250,
    ) -> tuple:
    """
    Compute ROC curve parameters and AUC score for binary or multiclass classification.

    Parameters
    ----------
    labels : array-like of shape (n_samples,)
        True class labels.
    outputs : array-like of shape (n_samples, n_classes)
        Model output probabilities for each class.
    mode : str, optional (default="binary")
        Type of classification task. Must be "binary" or "multiclass".
    classes : list, optional (default=list)
        List of class labels for multiclass classification. Required if mode is "multiclass".
    n_interp_points : int, optional (default=250)
        Number of points to interpolate the ROC curve at.

    Returns
    -------
    Tuple containing:
    - fpr: dict or array
        False positive rates. For binary classification, this is a 1D array.
        For multiclass classification, this is a dictionary with keys for each class
        and "macro", "micro", and "weighted" averages.
    - tpr: dict or array
        True positive rates. Same format as fpr.
    - thresholds: dict or array
        Thresholds used to compute fpr and tpr. Same format as fpr.
    - auc_score: dict or float
        AUC score. Same format as fpr.
    """
    assert mode in ["binary", "multiclass"], "mode must be 'binary' or 'multiclass'"

    if mode == "binary":
        probs = torch.sigmoid(torch.from_numpy(outputs)).numpy()
        fpr, tpr, thresholds = roc_curve(labels, probs[:, 1])
        auc_score = auc(fpr, tpr)

    else:
        probs = torch.softmax(torch.from_numpy(outputs), dim=1).numpy()
        labels_binarized = label_binarize(labels, classes=classes)

        support = np.sum(labels_binarized, axis=0)
        weights = support / np.sum(support)
        if np.any(support == 0):
            print("Warning: Some classes have no samples in the labels.\
                  This may lead to undefined ROC AUC scores for those classes.")

        fpr, tpr, thresholds, auc_score = {}, {}, {}, {}

        fpr["macro"] = fpr["weighted"] = np.linspace(0, 1, num=n_interp_points)
        tpr["macro"] = np.zeros_like(fpr["macro"])
        tpr["weighted"] = np.zeros_like(fpr["weighted"])

        for i in range(len(classes)):

            fpr[i], tpr[i], thresholds[i] = roc_curve(labels_binarized[:, i], probs[:, i])
            auc_score[i] = auc(fpr[i], tpr[i])
            # Interpolate TPR at common FPR points
            tpr["macro"] += np.interp(fpr["macro"], fpr[i], tpr[i])
            tpr["weighted"] += weights[i] * np.interp(fpr["weighted"], fpr[i], tpr[i])
        
        tpr["macro"] /= len(classes)

        fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(
            labels_binarized.ravel(),
            probs.ravel(),
        )
        tpr["micro"] = np.interp(fpr["macro"], fpr["micro"], tpr["micro"])
        thresholds["micro"] = np.interp(fpr["macro"], fpr["micro"], thresholds["micro"])
        fpr["micro"] = fpr["macro"]

        for key in ["macro", "micro", "weighted"]:
            auc_score[key] = roc_auc_score(labels, probs, average=key, multi_class="ovr")

    return fpr, tpr, thresholds, auc_score


def compute_multilabel_roc(
        y_true: np.ndarray,
        outputs: np.ndarray,
        classes: list,
        n_interp_points: int = 250,
    ) -> tuple:
    """
    Compute ROC curve parameters and AUC score for multilabel classification.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_classes)
        True class labels.
    outputs : array-like of shape (n_samples, n_classes)
        Model output logits for each class.
    classes : list
        List of class labels for multiclass classification.
    n_interp_points : int, optional (default=250)
        Number of points to interpolate the ROC curve at.

    Returns
    -------
    Tuple containing:
    - fpr: dict or array
        False positive rates. This is a dictionary with keys for each class
        and "macro", "micro", and "weighted" averages.
    - tpr: dict or array
        True positive rates. Same format as fpr.
    - thresholds: dict or array
        Thresholds used to compute fpr and tpr. Same format as fpr.
    - auc_score: dict or float
        AUC score. Same format as fpr.
    """
    probs = torch.sigmoid(torch.from_numpy(outputs)).numpy()

    support = np.sum(y_true, axis=0)
    weights = support / np.sum(support)
    if np.any(support == 0):
        print("Warning: Some classes have no samples in the labels.\
                This may lead to undefined ROC AUC scores for those classes.")

    fpr, tpr, thresholds, auc_score = {}, {}, {}, {}

    fpr["macro"] = fpr["weighted"] = np.linspace(0, 1, num=n_interp_points)
    tpr["macro"] = np.zeros_like(fpr["macro"])
    tpr["weighted"] = np.zeros_like(fpr["weighted"])

    for i, c in enumerate(classes):

        fpr[c], tpr[c], thresholds[i] = roc_curve(y_true[:, i], probs[:, i])
        auc_score[c] = auc(fpr[c], tpr[c])

        # Interpolate TPR at common FPR points
        tpr["macro"] += np.interp(fpr["macro"], fpr[c], tpr[c])
        tpr["weighted"] += weights[i] * np.interp(fpr["weighted"], fpr[c], tpr[c])

    tpr["macro"] /= len(classes)

    fpr["micro"], tpr["micro"], thresholds["micro"] = roc_curve(
        y_true.ravel(),
        probs.ravel(),
    )
    tpr["micro"] = np.interp(fpr["macro"], fpr["micro"], tpr["micro"])
    thresholds["micro"] = np.interp(fpr["macro"], fpr["micro"], thresholds["micro"])
    fpr["micro"] = fpr["macro"]

    for key in ["macro", "micro", "weighted"]:
        auc_score[key] = auc(fpr[key], tpr[key])

    return fpr, tpr, thresholds, auc_score


def multilabel_auc(y_true: np.ndarray, outputs: np.ndarray) -> float:
    """
    Compute the average AUC score for a multilabel classification problem.
    Adapted from https://github.com/MedMNIST/MedMNIST/
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_classes)
        True binary labels for each class.
    outputs : array-like of shape (n_samples, n_classes)
        Predicted logits for each class.
    
    Returns
    -------
    float
        Average AUC score across all classes.
    """

    assert y_true.shape == outputs.shape, "y_true and outputs must have the same shape"

    probs = torch.sigmoid(torch.from_numpy(outputs)).numpy()

    n_classes = y_true.shape[1]
    auc = 0
    for i in range(n_classes):
        label_auc = roc_auc_score(y_true[:, i], probs[:, i])
        auc += label_auc
    ret = auc / n_classes

    return ret


def multilabel_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the average accuracy score for a multilabel classification problem.
    Adapted from https://github.com/MedMNIST/MedMNIST/

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_classes)
        True binary labels for each class.
    y_pred : array-like of shape (n_samples, n_classes)
        Predicted one-hot labels for each sample (after thresholding).
    
    Returns
    -------
    float
        Average accuracy score across all classes.
    """

    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"

    n_classes = y_true.shape[1]
    acc = 0
    for label in range(n_classes):
        label_acc = accuracy_score(y_true[:, label], y_pred[:, label])
        acc += label_acc
    ret = acc / n_classes

    return ret
