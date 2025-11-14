"""
GradCAM Visualization for PyTorch.
"""

# Author: Atif Khurshid
# Created: 2025-11-14
# Modified: None
# Version: 1.0
# Changelog:
#     - 2025-11-14: Initial version.

from typing import Optional, Union

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.hooks import RemovableHandle


class GradCAMPyTorch(object):

    def __init__(self, model: nn.Module, target_layer: str):
        """
        Grad-CAM implementation for visualizing class activation maps.
        
        Parameters
        ----------
        model : nn.Module
            The neural network model.
        target_layer : str
            The name of the target layer to extract activations and gradients from.

        Attributes
        ----------
        activations : Optional[torch.Tensor]
            Stored activations from the forward pass.
        gradients : Optional[torch.Tensor]
            Stored gradients from the backward pass.
        handles : list[RemovableHandle]
            List of hook handles for removing hooks later.
        """
        super().__init__()
        
        self.model = model
        self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles: list[RemovableHandle] = []


    def get_gradcam_heatmap(
            self,
            x: torch.Tensor,
            *model_args: list,
            class_index: int | list[int],
            **model_kwargs: dict
        ) -> torch.Tensor:
        """
        Compute the Grad-CAM heatmap for the given input(s) and class index(s).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).
        class_index : int or list of int
            Target class index or list of class indices for each input in the batch.

        Returns
        -------
        heatmaps : torch.Tensor
            Grad-CAM heatmaps of shape (B, H, W).
        """
        output = self.model(x, *model_args, **model_kwargs)

        batch_size = x.shape[0]

        if isinstance(class_index, int):
            class_index = [class_index] * batch_size

        heatmaps = []

        for i in range(batch_size):

            self.model.zero_grad()  
            target = output[i, class_index[i]]
            target.backward(retain_graph=True)

            gradients = self.gradients[i] # Shape: (C, H, W)
            activations = self.activations[i]  # Shape: (C, H, W)

            feature_weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
            heatmap = torch.sum(feature_weights * activations, dim=0)
            heatmap = torch.relu(heatmap)

            heatmaps.append(heatmap)
        
        heatmaps = torch.stack(heatmaps, dim=0)  # Shape: (B, H, W)

        return heatmaps


    def visualize_heatmap(
            self,
            image: Union[torch.Tensor, np.ndarray],
            heatmap: torch.Tensor,
            normalize: bool = True,
            interpolation: str = 'bilinear',
            cmap: str = 'jet',
            alpha: float = 0.5
        ):
        """
        Overlay the Grad-CAM heatmap on the original image.

        Parameters
        ----------
        image : torch.Tensor
            The original image tensor of shape (3, H, W).
        heatmap : torch.Tensor
            The Grad-CAM heatmap tensor of shape (H, W).
        normalize : bool, optional
            Whether to normalize the heatmap before overlaying. Default is True.
        interpolation : str, optional
            The interpolation method for resizing the heatmap. Default is 'bilinear'.
        cmap : str, optional
            The colormap to use for the heatmap. Default is 'jet'.
        alpha : float, optional
            The transparency factor for overlaying the heatmap. Default is 0.5.

        Returns
        -------
        overlayed_image : torch.Tensor
            The image with the heatmap overlayed.
        """
        if isinstance(image, torch.Tensor):
            image = image.numpy()
            if image.ndim == 3:
                image = image.transpose(1, 2, 0)  # Shape: (H, W, 3)

        if np.issubdtype(image.dtype, np.integer):
            image = image.astype(np.float32) / 255.0

        if normalize:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-7)

        heatmap = F.interpolate(
            heatmap.unsqueeze(0).unsqueeze(0),  # Shape: (1, 1, H, W)
            size=image.shape[:2],
            mode=interpolation,
        ).squeeze(0).squeeze(0)  # Shape: (H, W)

        heatmap = plt.get_cmap(cmap)(heatmap.numpy())[:, :, :3]  # Apply colormap

        overlayed_image = alpha * heatmap + (1 - alpha) * image
        overlayed_image = np.clip(overlayed_image, 0, 1)

        return overlayed_image


    def _register_hooks(self):
        """
        Register forward and backward hooks on the target layer
        to store activations and gradients.
        """
        def forward_hook(module, input, output):
            self.activations = output.detach().clone().cpu()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach().clone().cpu()

        module = self.model.get_submodule(self.target_layer)
        self.handles.append(module.register_forward_hook(forward_hook))
        self.handles.append(module.register_full_backward_hook(backward_hook))


    def __enter__(self):
        """
        Context manager entry. Registers hooks on the target layer.
        """
        self._register_hooks()

        return self


    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context manager exit. Removes hooks and clears stored gradients and activations.
        """
        self.gradients = None
        self.activations = None

        for handle in self.handles:
            handle.remove()
        self.handles = []


    def __call__(self, *args, **kwargs):
        """
        Allows the instance to be called like a function to get the Grad-CAM heatmap.

        Parameters
        ----------
        *args : list
            Positional arguments to pass to `get_gradcam_heatmap`.
        **kwargs : dict
            Keyword arguments to pass to `get_gradcam_heatmap`.
        
            
        Returns
        -------
        torch.Tensor
            The Grad-CAM heatmap(s).
        """
        return self.get_gradcam_heatmap(*args, **kwargs)
    