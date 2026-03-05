"""
Utility functions for visualizations.
"""

# Author: Atif Khurshid
# Created: 2026-03-05
# Modified: None
# Version: 1.0
# Changelog:
#     - 2026-03-05: Added plt_save_and_show function.

from typing import Optional


def plt_save_and_show(
        plt,
        save_path: Optional[str] = None,
        dpi: int = 600,
        save_format: str = 'png',
        transparent: bool = False,
    ) -> None:
    """
    Save a matplotlib figure to a file.

    Parameters
    ----------
    plt : matplotlib.pyplot
        The matplotlib.pyplot module containing the current figure to save.
    save_path : str, optional
        The file path to save the figure to. If None, the figure will not be saved.
        Default is None.
    dpi : int, optional
        Dots per inch for the saved figure. Default is 600.
    save_format : str, optional
        Format to save the figure in (e.g., 'png', 'jpg', 'pdf'). Default is 'png'.
    transparent : bool, optional
        Whether to save the figure with a transparent background. Default is False.
    """
    if save_path is not None:
        if not save_path.lower().endswith(f".{save_format.lower()}"):
            save_path += f".{save_format}"
        plt.savefig(save_path, dpi=dpi, format=save_format, transparent=transparent)

    plt.show()
