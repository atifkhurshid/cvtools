"""
Evaluation utils for running experiments.
"""

# Author: Atif Khurshid
# Created: 2026-05-17
# Modified: 2026-05-26
# Version: 1.1
# Changelog:
#     - 2026-05-17: Initial version.
#     - 2026-05-26: Added error handling.

from typing import Optional

import wandb
import numpy as np
import pandas as pd

from wandb.plot.custom_chart import CustomChart
from ....metrics import compute_roc
from ....metrics import compute_multilabel_roc
from ....evaluation.classification import evaluate_classification
from ....evaluation.classification import evaluate_multilabel_classification


def evaluate_model(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        outputs: np.ndarray,
        classes: list,
        max_confusion_classes: int = 20,
        max_roc_classes: int = 16,
        run: Optional[wandb.Run] = None,
    ) -> dict:
    """
    Evaluate a classification model and log results.

    Parameters
    ----------
    y_true : np.ndarray
        True class labels, shape (n_samples,).
    y_pred : np.ndarray
        Predicted class labels, shape (n_samples,).
    outputs : np.ndarray
        Predicted probabilities for each class, shape (n_samples, n_classes).
    classes : list
        List of class names.
    max_confusion_classes : int, optional
        Maximum number of classes to log confusion matrix for. If the number of classes exceeds this,
        the confusion matrix will not be logged to Weights & Biases.
    max_roc_classes : int, optional
        Maximum number of classes to log individual ROC curves for. If the number of classes exceeds this,
        only micro, macro, and weighted ROC curves will be logged.
    run : wandb.Run, optional
        Weights & Biases run to log results to. If None, results will only be printed to console.
    
    Returns
    -------
    dict
        Summary of evaluation metrics (accuracy, precision, recall, f1-score, auc).
    """
    report = evaluate_classification(
        y_true, y_pred, outputs, class_names=classes,
        confusion=False, roc=False, print_report=False, return_report=True)

    summary = {
        "accuracy": report["accuracy"] * 100,
        "precision": report["weighted avg"]["precision"] * 100,
        "recall": report["weighted avg"]["recall"] * 100,
        "f1-score": report["weighted avg"]["f1-score"] * 100,
    }

    mode = "binary" if len(classes) == 2 else "multiclass"
    fpr, tpr, _, auc_scores = compute_roc(
        y_true, outputs, mode = mode, classes = classes)

    if mode == "binary":
        fpr = {"binary": fpr}
        tpr = {"binary": tpr}
        auc_scores = {"binary": auc_scores}
        summary["auc"] = auc_scores["binary"]
    else:
        summary["auc"] = auc_scores["weighted"]

    if len(classes) <= max_confusion_classes:
        try:
            confusion = wandb.plot.confusion_matrix(
                y_true = y_true,
                preds = y_pred,
                class_names = classes,
                title = "Confusion Matrix",
            )
        except Exception as e:
            print(f"Error logging summary to Weights & Biases: {e}")
    else:
        confusion = None

    report_evaluation_results(
        summary = summary,
        report = report,
        fpr = fpr,
        tpr = tpr,
        auc_scores = auc_scores,
        classes = classes,
        confusion = confusion,
        max_roc_classes = max_roc_classes,
        run = run,
    )

    return summary


def evaluate_multilabel_model(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        outputs: np.ndarray,
        classes: list,
        run: Optional[wandb.Run] = None,
    ) -> dict:
    """
    Evaluate a multilabel classification model and log results.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels for each class, shape (n_samples, n_classes).
    y_pred : np.ndarray
        Predicted binary labels for each class, shape (n_samples, n_classes).
    outputs : np.ndarray
        Predicted logits for each class, shape (n_samples, n_classes).
    classes : list
        List of class names.
    run : wandb.Run, optional
        Weights & Biases run to log results to. If None, results will only be printed to console.

    Returns
    -------
    dict
        Summary of evaluation metrics (accuracy, precision, recall, f1-score, auc).
    """
    report = evaluate_multilabel_classification(
        y_true, y_pred, outputs, class_names=classes,
        roc=False, print_report=False, return_report=True)

    fpr, tpr, _, auc_scores = compute_multilabel_roc(
        y_true, outputs, classes=classes)

    summary = {
        "accuracy": report["multilabel_accuracy"] * 100,
        "precision": report["weighted avg"]["precision"] * 100,
        "recall": report["weighted avg"]["recall"] * 100,
        "f1-score": report["weighted avg"]["f1-score"] * 100,
        "auc": report["multilabel_auc"],
    }

    report_evaluation_results(
        summary = summary,
        report = report,
        fpr = fpr,
        tpr = tpr,
        auc_scores = auc_scores,
        classes = classes,
        confusion = None,
        max_roc_classes = 16,
        run = run,
    )

    return summary


def report_evaluation_results(
        summary: dict,
        report: dict,
        fpr: dict,
        tpr: dict,
        auc_scores: dict,
        classes: list,
        confusion: Optional[CustomChart] = None,
        max_roc_classes: int = 16,
        run: Optional[wandb.Run] = None,
    ):
    """
    Helper function to log evaluation results to console and optionally to Weights & Biases.

    Parameters
    ----------
    summary : dict
        Summary of key evaluation metrics (accuracy, precision, recall, f1-score, auc).
    report : dict
        Detailed classification report (as returned by sklearn's classification_report).
    fpr : dict
        False positive rates for ROC curve, keyed by class and average types.
    tpr : dict
        True positive rates for ROC curve, keyed by class and average types.
    auc_scores : dict
        AUC scores for each class and average types.
    classes : list
        List of class names.
    confusion : wandb.Plot, optional
        Confusion matrix plot to log to Weights & Biases, if provided.
    max_roc_classes : int, optional
        Maximum number of classes to log individual ROC curves for. If the number of classes exceeds this,
        only micro, macro, and weighted ROC curves will be logged.
    run : wandb.Run, optional
        Weights & Biases run to log results to. If None, results will only be printed to console.
    """
    print("\nEvaluation Summary:")
    for metric, value in summary.items():
        print(f"\t{metric}: {value:.4f}")
    print()

    if run is not None:

        try:
            run.log({f"test/{k}": v for k, v in summary.items()})
        except Exception as e:
            print(f"Error logging summary to Weights & Biases: {e}")

        try:
            run.log({
                "test/classification_report": wandb.Table(
                    dataframe=pd.DataFrame(report).transpose().reset_index()
                ),
            })
        except Exception as e:
            print(f"Error logging classification report to Weights & Biases: {e}")

        try:
            if confusion is not None:
                run.log({"test/confusion_matrix": confusion})
        except Exception as e:
            print(f"Error logging confusion matrix to Weights & Biases: {e}")

        if len(classes) == 2:
            roc_titles = ["binary"]
        elif len(classes) <= max_roc_classes:
            roc_titles = [*classes, "micro", "macro", "weighted"]
        else:
            roc_titles = ["micro", "macro", "weighted"]

        try:
            df = pd.DataFrame(list(auc_scores.items()), columns=["Class", "AUC"])
            wandb.log({
                "test/auc_scores": wandb.Table(dataframe=df)
            })
        except Exception as e:
            print(f"Error logging AUC scores to Weights & Biases: {e}")

        for k in roc_titles:
            try:
                table = wandb.Table(
                    data = np.stack([fpr[k], tpr[k], [k]*len(fpr[k])], axis=1), columns=["fpr", "tpr", "class"]
                )
                wandb.log({
                    f"roc/roc_{k}": wandb.plot.plot_table(
                        data_table = table,
                        vega_spec_name="wandb/area-under-curve/v0",
                        fields = {
                            "x": "fpr",
                            "y": "tpr",
                            "class": "class",
                        },
                        string_fields = {
                            "title": f"ROC Curve - {k} (AUC={auc_scores[k]:.3f})",
                            "y": "True positive rate",
                            "x": "False positive rate",
                        },
                        split_table=True
                    )
                })
            except Exception as e:
                print(f"Error logging ROC curve for class '{k}' to Weights & Biases: {e}")
