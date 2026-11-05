"""
Bounding box visualization functions.
"""

# Author: Atif Khurshid
# Created: 2026-11-05
# Modified: None
# Version: 1.0
# Changelog:
#     - 2026-11-05: Added plot_boxes function.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_boxes(
    boxes: np.ndarray,
    figsize: tuple = (8, 8),
    edgecolors: str = "black",
    facecolor: str = "none",
    linewidths: float = 0.5,
    alpha: float = 0.7,
    equal_aspect: bool = True,
    invert_yaxis: bool = True,
    title = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot multiple bounding boxes on a blank canvas using matplotlib.PolyCollection.

    Parameters
    ----------
    boxes : np.ndarray
        Numpy array of shape (N, 4) containing bounding boxes in the format (x0, y0, x1, y1).
    figsize : tuple, optional
        Size of the figure. Default is (8, 8).
    edgecolors : str, optional
        Color of the edges of the boxes. Default is "black".
    facecolor : str, optional
        Color of the faces of the boxes. Default is "white".
    linewidths : float, optional
        Width of the edges of the boxes. Default is 0.5.
    alpha : float, optional
        Transparency level of the boxes. Default is 0.7.
    equal_aspect : bool, optional
        Whether to set equal aspect ratio for the axes. Default is True.
    invert_yaxis : bool, optional
        Whether to invert the y-axis to match image coordinates. Default is True.
    title : str, optional
        Title of the plot. Default is None.
    **kwargs : dict
        Additional keyword arguments for the PolyCollection.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object with the plotted boxes.
    """
    # Convert boxes from (N, 4) to (N, 4, 2) for PolyCollection
    x0, y0, x1, y1 = boxes.T
    verts = np.stack([
            np.stack([x0, y0], axis=1),
            np.stack([x1, y0], axis=1),
            np.stack([x1, y1], axis=1),
            np.stack([x0, y1], axis=1),
        ], axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.add_collection(
        PolyCollection(verts, facecolors=facecolor, edgecolors=edgecolors,
                       linewidths=linewidths, alpha=alpha, **kwargs)
    )

    if invert_yaxis:
        ax.invert_yaxis()

    if equal_aspect:
        ax.set_aspect("equal")

    if title is not None:
        ax.set_title(title)

    return ax


def plot_boxes_multicolor(
    boxes: np.ndarray,
    measure: str = "sqrt_area",
    cmap: str = "viridis",
    figsize: tuple = (8, 8),
    facecolor: str = "none",
    linewidths: float = 0.5,
    equal_aspect: bool = True,
    invert_yaxis: bool = True,
    title = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot multiple bounding boxes with different colors based on a measurement.

    Parameters
    ----------
    boxes : np.ndarray
        Numpy array of shape (N, 4) containing bounding boxes in the format (x0, y0, x1, y1).
    measure : str, optional
        Measurement to determine the color of each box.
        Options are "sqrt_area", "width", "height", "max_side", or "aspect".
    cmap : str, optional
        Colormap to use for the colors. Default is "viridis".
    figsize : tuple, optional
        Size of the figure. Default is (8, 8).
    edgecolors : str, optional
        Color of the edges of the boxes. Default is "black".
    facecolor : str, optional
        Color of the faces of the boxes. Default is "white".
    linewidths : float, optional
        Width of the edges of the boxes. Default is 0.5.
    equal_aspect : bool, optional
        Whether to set equal aspect ratio for the axes. Default is True.
    invert_yaxis : bool, optional
        Whether to invert the y-axis to match image coordinates. Default is True.
    title : str, optional
        Title of the plot. Default is None.
    **kwargs : dict
        Additional keyword arguments for the PolyCollection.
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object with the plotted boxes.
    """
    # Convert boxes from (N, 4) to (N, 4, 2) for PolyCollection
    x0, y0, x1, y1 = boxes.T

    widths, heights = x1 - x0, y1 - y0

    if measure == "sqrt_area":
        values = np.sqrt(widths * heights)
    elif measure == "width":
        values = widths
    elif measure == "height":
        values = heights
    elif measure == "max_side":
        values = np.maximum(widths, heights)
    elif measure == "aspect":
        values = np.log2(widths / heights)
    else:
        raise ValueError(f"Invalid measure: {measure}.")

    verts = np.stack([
            np.stack([x0, y0], axis=1),
            np.stack([x1, y0], axis=1),
            np.stack([x1, y1], axis=1),
            np.stack([x0, y1], axis=1),
        ], axis=1)

    # Draw small boxes last
    order = np.argsort(-values)

    fig, ax = plt.subplots(figsize=figsize)

    coll = PolyCollection(
        verts[order],
        facecolors=facecolor,
        array = values[order],
        cmap = cmap,
        norm = Normalize(vmin=values.min(), vmax=values.max()),
        linewidths=linewidths,
        **kwargs
    )
    ax.add_collection(coll)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(coll, cax=cax, label=measure)
    
    if invert_yaxis:
        ax.invert_yaxis()

    if equal_aspect:
        ax.set_aspect("equal")

    if title is not None:
        ax.set_title(title)

    return ax
