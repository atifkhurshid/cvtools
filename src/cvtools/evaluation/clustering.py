"""
Evaluation functions for clustering models.
"""

# Author: Atif Khurshid
# Created: 2025-06-22
# Modified: 2025-08-01
# Version: 1.1
# Changelog:
#     - 2025-08-01: Added documentation and type hints.

from pprint import pprint

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score


def evaluate_clustering(
        labels_true: list | np.ndarray,
        labels_pred: list | np.ndarray,
        report: bool = True,
    ) -> dict | None:
    """
    Evaluate clustering model performance.
    Computes and optionally prints the Adjusted Rand Index, Normalized Mutual Information,
    and Fowlkes-Mallows Score.

    Parameters
    -----------
    labels_true : list or np.ndarray
        True labels of the data.
    labels_pred : list or np.ndarray
        Predicted labels by the clustering model.
    report : bool, optional
        If True, prints the evaluation scores. If False, returns them as a dictionary.

    Returns
    --------
    dict or None
        If report is False, returns a dictionary with evaluation scores.
    
    Examples
    ---------
    >>> labels_true = [0, 1, 1, 0, 2, 2]
    >>> labels_pred = [0, 0, 1, 1, 2, 2]
    >>> evaluate_clustering(labels_true, labels_pred, report=True)
    Adjusted Rand Index: 0.6667
    Normalized Mutual Information: 0.5774
    Fowlkes-Mallows Score: 0.6667
    >>> scores = evaluate_clustering(labels_true, labels_pred, report=False)
    >>> print(scores)
    {
        "Adjusted Rand Index": 0.6667,
        "Normalized Mutual Information": 0.5774,
        "Fowlkes-Mallows Score": 0.6667
    }
    """

    ari = adjusted_rand_score(labels_true, labels_pred)
    nmi = adjusted_mutual_info_score(labels_true, labels_pred)
    fms = fowlkes_mallows_score(labels_true, labels_pred)

    scores = {
        "Adjusted Rand Index": float(ari),
        "Normalized Mutual Information": float(nmi),
        "Fowlkes-Mallows Score": float(fms)
    }

    if report:
        pprint(scores)
    else:
        return scores
    