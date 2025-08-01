"""
Evaluation functions for classification models.
"""

# Author: Atif Khurshid
# Created: 2025-06-22
# Modified: 2025-08-01
# Version: 1.1
# Changelog:
#     - 2025-08-01: Added documentation and type hints.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_classification(
        y_true: list | np.ndarray,
        y_pred: list | np.ndarray,
        class_names: list | None = None,
        figsize: tuple[int, int] = (10, 8)
    ):
    """
    Evaluate classification model performance.
    Prints the classification report and displays a confusion matrix.

    Parameters
    -----------
    y_true : list or np.ndarray
        True labels of the data.
    y_pred : list or np.ndarray
        Predicted labels by the model.
    class_names : list, optional
        Names of the classes. If None, uses integer labels.
    figsize : tuple, optional
        Size of the confusion matrix plot.
    
    Examples
    ---------
    >>> y_true = [0, 1, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 1]
    >>> evaluate_classification(y_true, y_pred, class_names=['Class 0', 'Class 1', 'Class 2'], figsize=(8, 6))
    This will print the classification report and display a confusion matrix for the given true and predicted labels.
    """

    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_true)))]

    print(classification_report(y_true, y_pred, target_names=class_names))

    confusion = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
