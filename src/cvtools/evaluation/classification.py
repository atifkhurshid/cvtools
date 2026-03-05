"""
Evaluation functions for classification models.
"""

# Author: Atif Khurshid
# Created: 2025-06-22
# Modified: 2026-03-05
# Version: 1.4
# Changelog:
#     - 2025-08-01: Added documentation and type hints.
#     - 2025-08-18: Improved confusion matrix plotting.
#     - 2025-10-16: Added save functionality for confusion matrix figure.
#     - 2026-03-05: Added ROC curve plotting.

from typing import Optional, Union

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from .metrics import compute_roc
from ..visualization import plot_roc_curves
from ..visualization import display_confusion_matrix


def evaluate_classification(
        y_true: Union[list, np.ndarray],
        y_pred: Union[list, np.ndarray],
        outputs: Optional[np.ndarray] = None,
        class_names: Optional[list] = None,
        confusion: bool = True,
        roc: bool = True,
        report: bool = False,
        figsize: tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
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
    outputs : np.ndarray, optional
        Model output logits for each class. Required if roc is True.
    class_names : list, optional
        Names of the classes. If None, uses integer labels.
    confusion : bool, optional
        Whether to display the confusion matrix. Default is True.
    roc : bool, optional
        Whether to compute and display ROC curves for each class. Default is True.
    report : bool, optional
        Whether to return the classification report as a dictionary. Default is False.
    figsize : tuple, optional
        Size of the confusion matrix plot. Default is (10, 8).
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
        display_confusion_matrix(
            confusion,
            class_names,
            figsize = figsize,
            save_path = save_path + "_confusion" if save_path is not None else None,
            save_dpi = save_dpi,
            save_format = save_format,
        )

    if roc:
        fpr, tpr, _, auc_scores = compute_roc(
            y_true,
            outputs,
            mode = "binary" if len(class_names) == 2 else "multiclass",
            classes = np.arange(len(class_names)),
        )
        if len(class_names) == 2:
            roc_titles = [f"{class_names[1]} vs {class_names[0]}"]
            fpr = [fpr]
            tpr = [tpr]
            auc_scores = [auc_scores]
        else:
            roc_titles = ["micro", "macro", "weighted"]
            fpr = [fpr[k] for k in roc_titles]
            tpr = [tpr[k] for k in roc_titles]
            auc_scores = [auc_scores[k] for k in roc_titles]

        plot_roc_curves(
            fpr,
            tpr,
            auc_scores,
            labels = roc_titles,
            figsize = figsize,
            save_path = save_path + "_roc" if save_path is not None else None,
            save_dpi = save_dpi,
            save_format = save_format,
        )

    if report:
        report = classification_report(
            y_true, y_pred, target_names=class_names, digits=digits,
            zero_division=zero_division, output_dict=True, **kwargs)
        
        return report
