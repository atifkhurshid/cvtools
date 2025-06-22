"""
Evaluation functions for clustering models.
"""

# Author: Atif Khurshid
# Created: 2025-06-22
# Modified: None
# Version: 1.0
# Changelog:
#     - None

import numpy as np
from pprint import pprint
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import fowlkes_mallows_score


def evaluate_clustering(
        labels_true: list | np.ndarray,
        labels_pred: list | np.ndarray,
        report: bool = True,
    ) -> dict | None:

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
    