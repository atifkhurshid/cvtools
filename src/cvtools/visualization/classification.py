"""
Visualizations for classification.
"""

# Author: Atif Khurshid
# Created: 2026-03-05
# Modified: 2026-03-13
# Version: 1.1
# Changelog:
#     - 2026-03-05: Added ROC curve plotting functions.
#     - 2026-03-05: Added confusion matrix display function.
#     - 2026-03-13: Added option to plot diagonal line in ROC curve functions.

from typing import Union, Optional

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from .utils import plt_save_and_show


def display_confusion_matrix(
        confusion: np.ndarray,
        class_names: list,
        figsize: tuple[int, int] = (10, 8),
        save_path: Optional[str] = None,
        save_dpi: int = 600,
        save_format: str = 'png',
    ) -> None:

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    plt_save_and_show(plt, save_path, save_dpi, save_format)


def plot_roc_curve(
        fpr: list,
        tpr: list,
        auc_score: float,
        label: str = "ROC Curve",
        color: str = "tab-blue",
        linestyle: str = "-",
        diagonal: bool = False,
        figsize: tuple = (7, 6),
        facecolor: str = "none",
        save_path: Union[str, None] = None,
        save_dpi: int = 600,
        save_format: str = 'png',
    ) -> None:
    """
    Plot ROC curve from given FPR and TPR values.

    Parameters
    ----------
    fpr : list
        False positive rates.
    tpr : list
        True positive rates.
    auc_score : float
        AUC score to display in the legend.
    label : str, optional
        Label for the ROC curve in the legend. Default is "ROC Curve".
    color : str, optional
        Color of the ROC curve. Default is "tab-blue".
    linestyle : str, optional
        Line style for the ROC curve. Default is "-".
    diagonal : bool, optional
        Whether to plot the diagonal line for reference. Default is False.
    figsize : tuple, optional
        Size of the figure. Default is (7, 6).
    facecolor : str, optional
        Background color of the plot. Default is "none".
    save_path : str or None, optional
        Path to save the figure, if None the figure is not saved. Default is None.
    save_dpi : int, optional
        Dots per inch for saving the figure. Default is 600.
    save_format : str, optional
        Format to save the figure. Default is 'png'.
    """
    fig, ax = plt.subplots(facecolor=facecolor, figsize=figsize)

    # Plot the diagonal line for reference
    if diagonal:
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")

    ax.plot(fpr, tpr, label=f"{label} [{auc_score:.3f}]",
            color=color, linestyle=linestyle)
    
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    ax.legend(loc="lower right", frameon=False)

    ax.patch.set_facecolor(facecolor)

    plt_save_and_show(plt, save_path, save_dpi, save_format, transparent=True)


def plot_roc_curves(
        fprs: list[list],
        tprs: list[list],
        auc_scores: list[float],
        labels: list[str],
        colors: Union[list[str], str] = "auto",
        linestyles: Union[list[str], str] = "-",
        diagonal: bool = False,
        figsize: tuple = (7, 6),
        facecolor: str = "none",
        save_path: Union[str, None] = None,
        save_dpi: int = 600,
        save_format: str = 'png',
    ) -> None:
    """
    Plot multiple ROC curves on the same figure.

    Parameters
    ----------
    fprs : list[list]
        List of false positive rates for each plot.
    tprs : list[list]
        List of true positive rates for each plot.
    auc_scores : list[float]
        List of AUC scores for each plot.
    labels : list[str]
        List of labels for each plot, in the same order as fprs and tprs.
    colors : list[str] or str, optional
        List of colors for each plot or a single color for all plots. Default is "auto",
        which uses the tab10 color cycle.
    linestyles : list[str] or str, optional
        List of line styles for each plot or a single line style for all plots.
        Default is "-".
    diagonal : bool, optional
        Whether to plot the diagonal line for reference. Default is False.
    figsize : tuple, optional
        Size of the figure. Default is (7, 6).
    facecolor : str, optional
        Background color of the plot. Default is "none".
    save_path : str or None, optional
        Path to save the figure, if None the figure is not saved. Default is None.
    save_dpi : int, optional
        Dots per inch for saving the figure. Default is 600.
    save_format : str, optional
        Format to save the figure. Default is 'png'.
    """
    fig, ax = plt.subplots(facecolor=facecolor, figsize=figsize)

    # Plot the diagonal line for reference
    if diagonal:
        ax.plot([0, 1], [0, 1], color="gray", linestyle="--")

    if colors == "auto":
        colors = list(plt.get_cmap("tab10").colors)
        if len(labels) > len(colors):
            colors = colors * (len(labels) // len(colors) + 1)

    if isinstance(linestyles, str):
        linestyles = [linestyles] * len(labels)

    for i in range(len(fprs)):
        ax.plot(fprs[i], tprs[i], label=f"{labels[i]} [{auc_scores[i]:.3f}]",
                color=colors[i], linestyle=linestyles[i])

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    ax.legend(loc="lower right", frameon=False)

    ax.patch.set_facecolor(facecolor)

    plt_save_and_show(plt, save_path, save_dpi, save_format, transparent=True)
