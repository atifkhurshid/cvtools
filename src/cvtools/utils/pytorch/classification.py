"""
Utility functions for classification models.
"""

# Author: Atif Khurshid
# Created: 2025-11-11
# Modified: None
# Version: 1.0
# Changelog:
#     - 2025-11-11: Added topk_predictions function.

from typing import Optional

import torch
import numpy as np


def topk_predictions(
        output: torch.Tensor,
        k: int = 5,
        classes: Optional[list[str]] = None,
        **kwargs: dict,
    ) -> list[list[tuple[str, float]]]:
    """
    Get the top-k predictions from the model output.

    Parameters
    ----------
    output : torch.Tensor
        The model output tensor of shape (batch_size, n_classes).
    k : int, optional
        The number of top predictions to return, by default 5.
    classes : list[str], optional
        The list of class names corresponding to the output indices, by default None.
    **kwargs : dict
        Additional keyword arguments to pass to `torch.topk`.
    
    Returns
    -------
    list[list[tuple[str, float]]]
        A list of lists containing tuples of (class_name, score) for the top-k predictions
        for each sample in the batch.
    """
    res = torch.topk(output, k=k, **kwargs)
    scores_list = res.values.detach().cpu().numpy()
    labels_list = res.indices.detach().cpu().numpy()
    if classes is not None:
        labels_list = [[classes[idx] for idx in pred] for pred in labels_list.tolist()]

    predictions = [
        [(label, score.item()) for label, score in zip(labels, scores)]
        for labels, scores in zip(labels_list, scores_list)
    ]

    return predictions
