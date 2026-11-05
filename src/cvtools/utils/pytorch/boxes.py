"""
Utility functions for bounding boxes.
"""

# Author: Atif Khurshid
# Created: 2026-11-05
# Modified: None
# Version: 1.0
# Changelog:
#     - 2026-11-05: Added suppress_windows function.

import torch
from torchvision.ops import nms


def suppress_windows(
    x0: torch.Tensor,
    y0: torch.Tensor,
    x1: torch.Tensor,
    y1: torch.Tensor,
    iou_threshold: float = 0.5,
    prefer: str = "large",
):
    """
    Suppress overlapping windows using Non-Maximum Suppression (NMS).

    Parameters
    ----------
    x0 : torch.Tensor
        Tensor of x-coordinates of the top-left corner of the boxes.
    y0 : torch.Tensor
        Tensor of y-coordinates of the top-left corner of the boxes.
    x1 : torch.Tensor
        Tensor of x-coordinates of the bottom-right corner of the boxes.
    y1 : torch.Tensor
        Tensor of y-coordinates of the bottom-right corner of the boxes.
    iou_threshold : float, optional
        Intersection over Union (IoU) threshold for NMS. Default is 0.5.
    prefer : str, optional
        Preference for keeping boxes with larger or smaller areas. Options are "large" or "small".
    
    Returns
    -------
    kept_boxes : torch.Tensor
        Tensor of boxes that are kept after NMS, in the format (x0, y0, x1, y1).
    kept_idx : torch.Tensor
        Tensor of indices of the boxes that are kept after NMS.
    """
    boxes = torch.stack(
        [x0.reshape(-1), y0.reshape(-1), x1.reshape(-1), y1.reshape(-1)], dim=1
    ).float()
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    scores = area if prefer == "large" else -area
    keep = nms(boxes, scores, iou_threshold)

    return boxes[keep], keep
    