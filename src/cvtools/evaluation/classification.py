"""
Evaluation functions for classification models.
"""

# Author: Atif Khurshid
# Created: 2025-06-22
# Modified: 2025-10-16
# Version: 1.3
# Changelog:
#     - 2025-08-01: Added documentation and type hints.
#     - 2025-08-18: Improved confusion matrix plotting.
#     - 2025-10-16: Added save functionality for confusion matrix figure.

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_classification(
        y_true: list | np.ndarray,
        y_pred: list | np.ndarray,
        class_names: list | None = None,
        confusion: bool = True,
        figsize: tuple[int, int] = (10, 8),
        report: bool = False,
        save_path: str | None = None,
        save_dpi: int = 600,
        save_format: str = 'png',
        digits: int = 4,
        zero_division: int = 0,
        **kwargs: dict,
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
    confusion : bool, optional
        Whether to display the confusion matrix. Default is True.
    figsize : tuple, optional
        Size of the confusion matrix plot.
    report : bool, optional
        Whether to return the classification report as a dictionary. Default is False.
    save_path : str | None, optional
        Path to save the figure, if None the figure is not saved, default is None.
    save_dpi : int, optional
        Dots per inch for saving the figure, default is 600.
    save_format : str, optional
        Format to save the figure, default is 'png'.
    digits : int, optional
        Number of decimal places for formatting in the report. Default is 4.
    zero_division : int, optional
        Sets the value to return when there is a zero division. Default is 0.
    **kwargs : dict
        Additional keyword arguments for sklearn's classification_report.
    
    Returns
    -------
    dict or None
        If report is True, returns the classification report as a dictionary.
    
    Examples
    ---------
    >>> y_true = [0, 1, 2, 2, 0, 1]
    >>> y_pred = [0, 0, 2, 2, 0, 1]
    >>> evaluate_classification(y_true, y_pred, class_names=['Class 0', 'Class 1', 'Class 2'], figsize=(8, 6))
    This will print the classification report and display a confusion matrix for the given true and predicted labels.
    """

    if class_names is None:
        class_names = [str(i) for i in range(len(np.unique(y_true)))]

    print(classification_report(
        y_true, y_pred, target_names=class_names, digits=digits, zero_division=zero_division, **kwargs))

    if confusion:
        confusion = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=figsize)
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        if save_path is not None:
            plt.savefig(save_path, dpi=save_dpi, format=save_format)
        plt.show()

    if report:
        report = classification_report(
            y_true, y_pred, target_names=class_names, digits=digits,
            zero_division=zero_division, output_dict=True, **kwargs)
        
        return report
    