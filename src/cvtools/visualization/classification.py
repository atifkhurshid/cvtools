"""
Visualizations for classification.
"""

# Author: Atif Khurshid
# Created: 2026-03-05
# Modified: 2026-03-19
# Version: 1.2
# Changelog:
#     - 2026-03-05: Added ROC curve plotting functions.
#     - 2026-03-05: Added confusion matrix display function.
#     - 2026-03-13: Added option to plot diagonal line in ROC curve functions.
#     - 2026-03-19: Added function to plot mean ROC curve with confidence intervals.

from typing import Union, Optional

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import norm

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


def plot_roc_curves_with_confidence_intervals(
        fprs_list: list[list],
        tprs_list: list[list],
        aucs_list: list[float],
        labels: list[str],
        colors: Union[list[str], str] = "auto",
        confidence: float = 0.95,
        n_points: int = 250,
        title: str = "ROC Curves with Confidence Intervals",
        figsize: tuple = (10, 8),
        facecolor: str = "none",
        save_path: Union[str, None] = None,
        save_dpi: int = 600,
        save_format: str = 'png',
    ):
    """
    Plot mean ROC curves with confidence intervals for multiple experiments,
    each with multiple runs.

    Parameters
    ----------
    fprs_list : list[list]
        List of lists of false positive rates for each trial.
    tprs_list : list[list]
        List of lists of true positive rates for each trial.
    aucs_list : list[float]
        List of AUC scores for each trial.
    labels : list[str]
        List of labels for each plot, in the same order as fprs_list and tprs_list.
    colors : list[str] or str, optional
        List of colors for each plot or a single color for all plots. Default is "auto",
        which uses the tab10 color cycle.
    confidence : float, optional
        Confidence level for the confidence interval. Default is 0.95.
    n_points : int, optional
        Number of points to interpolate the mean ROC curve. Default is 250.
    title : str, optional
        Title of the plot. Default is "ROC Curve with Confidence Interval".
    figsize : tuple, optional
        Size of the figure. Default is (10, 8).
    facecolor : str, optional
        Background color of the plot. Default is "none".
    save_path : str or None, optional
        Path to save the figure, if None the figure is not saved. Default is None.
    save_dpi : int, optional
        Dots per inch for saving the figure. Default is 600.
    save_format : str, optional
        Format to save the figure. Default is 'png'.
    """

    fig, ax = plt.subplots(figsize=figsize)

    mean_fpr = np.linspace(0, 1, n_points)

    if colors == "auto":
        colors = list(plt.get_cmap("tab10").colors)
        if len(labels) > len(colors):
            colors = colors * (len(labels) // len(colors) + 1)

    for i, (fprs, tprs, aucs, label) in enumerate(zip(
        fprs_list, tprs_list, aucs_list, labels)):

        tprs_interp = []

        for fpr, tpr in zip(fprs, tprs):

            ax.plot(fpr, tpr, alpha=0.4, color=colors[i], lw=1)

            tpr_interp = np.interp(mean_fpr, fpr, tpr)
            tpr_interp[0] = 0.0
            tprs_interp.append(tpr_interp)

        tprs = np.array(tprs_interp)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        ax.plot(mean_fpr, mean_tpr, color=colors[i], lw=2,
                label=f"{label} (AUC = {mean_auc:.3f} ± {std_auc:.2f})")

        n_trials = len(tprs)
        z = norm.ppf((1 + confidence) / 2)
        tprs_upper = np.clip(mean_tpr + z * (std_tpr / np.sqrt(n_trials)), 0, 1)
        tprs_lower = np.clip(mean_tpr - z * (std_tpr / np.sqrt(n_trials)), 0, 1)

        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[i], alpha=0.2)

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        xlim=[0, 1],
        ylim=[0, 1],
        title=title
    )
    ax.legend(loc="lower right")
    ax.patch.set_facecolor(facecolor)

    plt.tight_layout()

    plt_save_and_show(plt, save_path, save_dpi, save_format, transparent=True)
