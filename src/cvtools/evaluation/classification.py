"""
Evaluation functions for classification models.
"""

# Author: Atif Khurshid
# Created: 2025-06-22
# Modified: None
# Version: 1.0
# Changelog:
#     - None

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
