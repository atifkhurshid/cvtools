"""
Eccentricity-dependent Pooling Layer

Adapted from https://github.com/kreimanlab/VisualSearchAsymmetry
"""

# Author: Atif Khurshid
# Created: 2026-04-24
# Modified: None
# Version: 1.0
# Changelog:
#    - 2026-04-24: Initial implementation.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from skimage.draw import disk


class CenterDependentPool2D(nn.Module):

    def __init__(
        self,
        input_size: int,
        rf_min: int = 2,
        ecc_slope: float = 0.2,
        deg2px: int = 30,
        fovea_size: float = 4,
        rf_quant: float = 1,
        stride: int = 2,
        pool_type: str = 'max',
    ):
        """
        Eccentricity-dependent pooling layer for 2D spatial data.

        Adapted from https://github.com/kreimanlab/VisualSearchAsymmetry

        Constructs concentric annular binary masks, each associated with a
        pooling window size scaled by eccentricity. The output is the sum of
        element-wise products: sum_i(Mask[i] * Pool_i(x)).

        Parameters
        ----------
        input_size : int
            Spatial size of the input feature map i.e. height = width = input_size.
        rf_min : int, optional
            Minimum receptive field size, by default 2.
        ecc_slope : float, optional
            Slope of the eccentricity-dependent scaling, by default 0.2.
        deg2px : int, optional
            Conversion factor from degrees to pixels, by default 30.
        fovea_size : float, optional
            Size of the foveal region in degrees, by default 4.
        rf_quant : float, optional
            Quantization factor for receptive field sizes, by default 1.
        stride : int, optional
            Stride for pooling operations, by default 2.
        pool_type : str, optional
            Type of pooling ('max' or 'avg'), by default 'max'.
        """
        super().__init__()

        self.input_size = input_size
        self.rf_min = rf_min
        self.ecc_slope = ecc_slope
        self.deg2px = deg2px
        self.fovea_size = fovea_size
        self.rf_quant = rf_quant
        self.stride = stride

        self.output_size = (input_size + 1) // stride if stride > 1 else input_size

        if pool_type == 'max':
            self.pool_fn = F.max_pool2d
        elif pool_type == 'avg':
            self.pool_fn = F.avg_pool2d
        else:
            raise ValueError("pool_type must be 'max' or 'avg'")

        self.masks: nn.Parameter
        self.rf_sizes: list[int]
        self.ecc_borders: list[int]
        self._build()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying eccentricity-dependent pooling.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).
        
        Returns
        -------
        torch.Tensor
            Eccentricity-pooled tensor of shape (B, C, H', W').
        """
        out = torch.zeros_like(x[:, :, :self.output_size, :self.output_size])

        for i, rf in enumerate(self.rf_sizes):
            
            k = max(rf, 1)
            
            pad_h = (k - 1) / 2
            pad_w = (k - 1) / 2
            pad_top = int(np.floor(pad_h))
            pad_bottom = int(np.ceil(pad_h))
            pad_left = int(np.floor(pad_w))
            pad_right = int(np.ceil(pad_w))

            x_padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')

            pooled = self.pool_fn(x_padded, kernel_size=k, stride=self.stride)

            mask = self.masks[i]

            out += pooled * mask

        return out


    def _build(self):
        """
        Construct masks and rf_sizes given spatial dimensions of input feature map.
        """
        mask_shape = (1, 1, self.output_size, self.output_size)
        center = self.output_size // 2

        ecc = round((self.fovea_size * self.deg2px) / 2)
        ecc_step = round((self.rf_quant * self.deg2px) / 2)
        fovea_size_px = self.fovea_size * self.deg2px

        self.masks = []
        self.rf_sizes = []
        self.ecc_borders = []

        if ecc > self.output_size / 2:

            mask = torch.ones(mask_shape, dtype=torch.float32)
            self.masks.append(mask)
            self.ecc_borders.append(2 * ecc)
            self.rf_sizes.append(self.rf_min)

        else:
            # First Mask: Foveal disk
            mask = torch.zeros(mask_shape, dtype=torch.float32)
            rr, cc = disk((center, center), ecc, shape=mask_shape[2:])
            mask[:, :, rr, cc] = 1

            self.masks.append(mask)
            self.ecc_borders.append(2 * ecc)
            self.rf_sizes.append(self.rf_min)

            ecc += ecc_step

            # Mid Masks: Annular regions
            while ecc < self.output_size / 2:
                mask = torch.zeros(mask_shape, dtype=torch.float32)
                rr, cc = disk((center, center), ecc, shape=mask_shape[2:])
                mask[:, :, rr, cc] = 1

                rr, cc = disk((center, center), ecc - ecc_step, shape=mask_shape[2:])
                mask[:, :, rr, cc] = 0

                self.masks.append(mask)
                self.ecc_borders.append(2 * ecc)
                self.rf_sizes.append(self.rf_min + round(self.ecc_slope * (ecc * 2 - fovea_size_px)))

                ecc += ecc_step
            
            # Final Mask: Peripheral remainder
            mask = torch.ones(mask_shape, dtype=torch.float32)
            rr, cc = disk((center, center), ecc - ecc_step, shape=mask_shape[2:])
            mask[:, :, rr, cc] = 0

            self.masks.append(mask)
            self.ecc_borders.append(2 * ecc)
            self.rf_sizes.append(self.rf_min + round(self.ecc_slope * (ecc * 2 - fovea_size_px)))

        self.masks = nn.Parameter(torch.stack(self.masks), requires_grad=False)


    def visualize_masks(
            self,
            cmap: str = 'viridis_r',
            figsize: tuple[int, int] = (6, 6)
        ) -> None:
        """
        Utility to visualize the generated masks.

        Parameters
        ----------
        cmap : str, optional
            Matplotlib colormap to use for visualization, by default 'viridis_r'.
        figsize : tuple[int, int], optional
            Size of the figure for visualization, by default (6, 6).
        """
        mask_image = torch.stack([rf * mask for rf, mask in zip(self.rf_sizes, self.masks)], dim=0)
        mask_image = mask_image.sum(dim=[0, 1, 2])

        cmap_ticks = np.arange(min(self.rf_sizes), max(self.rf_sizes)+1, step=1)

        cmap = plt.get_cmap(cmap, len(cmap_ticks))

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(mask_image.cpu(), cmap=cmap)

        cbar = fig.colorbar(
            im,
            ax=ax,
            ticks=cmap_ticks,
            fraction=0.046,
            pad=0.04,
            label='Receptive Field Size (pixels)'
        )

        plt.axis('off')
        plt.tight_layout()
        plt.show()
