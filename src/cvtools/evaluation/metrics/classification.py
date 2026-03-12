"""
Metrics for evaluating classification models.
"""

# Author: Atif Khurshid
# Created: 2026-03-05
# Modified: 2026-03-12
# Version: 1.1
# Changelog:
#     - 2026-03-05: Added ROC curve and AUC score computation.
#     - 2026-03-12: Added n_interp_points parameter to compute_roc function.
#     - 2026-03-12: Added interpolation of roc curve for micro weighted average.

from typing import Optional

import torch
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score, auc


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

        auc_score["macro"] = roc_auc_score(labels, probs, average="macro", multi_class="ovr")
        auc_score["micro"] = roc_auc_score(labels, probs, average="micro", multi_class="ovr")
        auc_score["weighted"] = roc_auc_score(labels, probs, average="weighted", multi_class="ovr")

    return fpr, tpr, thresholds, auc_score
