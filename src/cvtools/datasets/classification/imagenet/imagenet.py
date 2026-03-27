"""
ImageNet dataset for classification tasks.
"""

# Author: Atif Khurshid
# Created: 2026-03-26
# Modified: 2026-03-27
# Version: 1.0
# Changelog:
#     - 2026-03-26: Created ImageNet dataset class.
#     - 2026-03-27: Refactored code to match updated base class.

import os
from pathlib import Path
from typing import Optional, Union

import torch
from torchvision.datasets.utils import check_integrity

from ..generic import ClassificationDataset


class ImageNetDataset(ClassificationDataset):
    """
    A dataset class for loading the ImageNet dataset for classification tasks.
    """

    def __init__(
            self,
            root_dir: str,
            split: str = 'train',
            exts: Union[list[str], str] = "auto",
            hdf5_mode: Optional[str] = None,
            image_mode: Union[str, int] = 'RGB',
            image_scale: Optional[float] = None,
            image_size: Optional[Union[int, tuple[int, int]]] = None,
            preserve_aspect_ratio: bool = True,
            interpolation: Optional[int] = None,
        ):
        """
        Initializes the ImageNet dataset.

        Root directory must contain the meta.bin file, along with a train/val subdirectory 
        containing the images organized in class-specific subdirectories.

        Parameters
        ----------
        root_dir : str
            Path to the root directory containing class subdirectories.
        exts : list[str] | str, optional
            List of file extensions to consider as valid images. Default is "auto" (uses IMAGE_EXTENSIONS).
        hdf5_mode : str, optional
            If "stream", load images from an HDF5 file on-the-fly.
            If "preload", preload all images from the HDF5 file into memory. Default is None (load from files).
        image_mode : str | int, optional
            Mode to read images. Can be 'RGB', 'GRAY', or a cv2.IMREAD_... flag. Default is 'RGB'.
        image_scale : float, optional
            Scale factor to resize images. Default is None (no scaling).
        image_size : int | tuple, optional
            Size of the images to be resized to. If int, resizes the maximum dimension to this size.
            If tuple, should be (height, width). Default is None (no resizing).
        preserve_aspect_ratio : bool, optional
            If True, preserve the aspect ratio of the images when resizing. Default is True.
        interpolation : int, optional
            Interpolation method to use when resizing images. Default is None (uses default interpolation).
        
        Attributes
        ----------
        wnids : list[str]
            List of WordNet IDs corresponding to the classes in the dataset.
        """
        images_dir = os.path.join(root_dir, split)

        super().__init__(
            root_dir=images_dir,
            exts=exts,
            image_mode=image_mode,
            hdf5_mode=hdf5_mode,
            image_scale=image_scale,
            image_size=image_size,
            preserve_aspect_ratio=preserve_aspect_ratio,
            interpolation=interpolation,
        )

        self._initialize()

        # Reused from torchvision ImageNet dataset implementation
        wnid_to_classes = load_meta_file(root_dir)[0]
        self.wnids = self.classes
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]

        self.wnid2class = {wnid: c for wnid, c in zip(self.wnids, self.classes)}
        self.class2wnid = {v: k for k, v in self.wnid2class.items()}
        self.index2class = {i: c for i, c in enumerate(self.classes)}


    def wnid_to_class_name(self, wnid: str) -> str:
        """
        Convert a WordNet ID (wnid) to a human-readable class name.

        Parameters
        ----------
        wnid : str
            The WordNet ID to convert.

        Returns
        -------
        str
            The corresponding class name.
        """
        return self.wnid2class[wnid]


    def class_name_to_wnid(self, class_name: str) -> str:
        """
        Convert a human-readable class name to a WordNet ID (wnid).

        Parameters
        ----------
        class_name : str
            The class name to convert.

        Returns
        -------
        str
            The corresponding WordNet ID.
        """
        return self.class2wnid[class_name]


def load_meta_file(
        root: Union[str, Path],
        file: Optional[str] = None
    ) -> tuple[dict[str, str], list[str]]:
    """
    Reused from torchvision ImageNet dataset implementation
    """
    if file is None:
        file = "meta.bin"

    file = os.path.join(root, file)

    if check_integrity(file):
        return torch.load(file, weights_only=True)
    
    else:
        msg = (
            "The meta file {} is not present in the root directory or is corrupted. "
            "This file is automatically created by the ImageNet dataset."
        )
        raise RuntimeError(msg.format(file, root))
    