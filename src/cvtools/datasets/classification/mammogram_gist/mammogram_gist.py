"""
Dataloader for Mammogram dataset from https://github.com/gistmammocnn/MammogramGist-CNN
"""

# Author: Atif Khurshid
# Created: 2026-09-24
# Modified: None
# Version: 1.0
# Changelog:
#     - 2026-09-24: Initial version.

import os
import glob
from typing import Optional, Union

import numpy as np
import pandas as pd

from .._base import _ClassificationBaseImage


class MammogramGistDataset(_ClassificationBaseImage):

    def __init__(
        self,
        root_dir: str,
        preprocessing: str = "none",
        train: bool = True,
        image_mode: str = "GRAY",
        image_scale: Optional[float] = None,
        image_size: Optional[Union[int, tuple[int, int]]] = None,
        preserve_aspect_ratio: bool = True,
        interpolation: Optional[int] = None,
    ):
        """
        Mammogram dataset from Wurster et al., "Human Gist Processing Augments Deep Learning Breast Cancer Risk Assessment". 2019.
        Downloaded from https://github.com/gistmammocnn/MammogramGist-CNN

        The dataloader requires the original Images and the RadiologistData folder to be present in the root_dir.
        The images should be organized in subdirectories named after their class labels.
        This is already the case for Preprocessing2, Preprocessing3, and Preprocessing4.
        For Preprocessing1, the "Contralateral" folder is not present and should be copied in
        from the "Original Images" folder. The class folders should also be renamed to
        "Cancer", "Contralateral", and "Normal" to match the other preprocessing folders.

        Parameters
        ----------
        root_dir : str
            Path to the root directory of the dataset.
        preprocessing : str, optional
            The type of preprocessing to apply to the images. Default is "none".
            Options are "none", "same-direction", "crop", and "crop-same-direction".
        train : bool, optional
            If True, uses the training set. Default is True.
        image_mode : str, optional
            Mode to read images. Default is "GRAY" for grayscale images.
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
        images_dir : str
            Path to the directory containing the images.
        data : pd.DataFrame
            DataFrame containing the annotations and labels.
        classes : list
            List of unique class labels in the dataset.
        label2idx : dict
            Mapping from class labels to indices.
        idx2label : dict
            Mapping from indices to class labels.
        """
        super().__init__(
            root_dir=root_dir,
            image_mode=image_mode,
            image_scale=image_scale,
            image_size=image_size,
            preserve_aspect_ratio=preserve_aspect_ratio,
            interpolation=interpolation
        )
        self.root_dir = root_dir

        self.images_path = os.path.join(root_dir, "Images")
        if preprocessing == "none":
            self.images_path = os.path.join(self.images_path, "Preprocessing1")
        elif preprocessing == "same-direction":
            self.images_path = os.path.join(self.images_path, "Preprocessing2")
        elif preprocessing == "crop":
            self.images_path = os.path.join(self.images_path, "Preprocessing3")
        elif preprocessing == "crop-same-direction":
            self.images_path = os.path.join(self.images_path, "Preprocessing4")
        else:
            raise ValueError(f"Invalid preprocessing option: {preprocessing}. "
                             f"Valid options are 'none', 'same-direction', 'crop', and 'crop-same-direction'.")

        image_files_info = {
            "ImageID": [],
            "Extension": [],
            "Class": [],
        }

        for class_name in os.listdir(self.images_path):
            class_dir = os.path.join(self.images_path, class_name)
            if not os.path.isdir(class_dir):
                continue
            for image_filename in os.listdir(class_dir):
                if not image_filename.lower().endswith((".bmp", ".png")):
                    continue
                image_name, image_ext = os.path.splitext(image_filename)
                image_files_info["ImageID"].append(image_name)
                image_files_info["Extension"].append(image_ext)
                image_files_info["Class"].append(class_name)

        image_files_df = pd.DataFrame(image_files_info)

        annotations_df = pd.read_csv(
            os.path.join(self.root_dir, "RadiologistData", "radiologistInput.csv"))
        annotations_df.rename(columns={"Image Number in Database": "ImageID"}, inplace=True)
        annotations_df["ImageID"] = annotations_df["ImageID"].str.replace("-", "_")
        annotations_df = annotations_df.dropna().reset_index(drop=True)

        self.data = pd.merge(image_files_df, annotations_df, on="ImageID", how="outer")

        no_human_annotations = self.data["AvgResponseRating"].isna()
        if train:
            self.data = self.data[no_human_annotations].reset_index(drop=True)
            self.ratings = None
        else:
            # Use only images with human annotations for testing
            self.data = self.data[~no_human_annotations].reset_index(drop=True)
            self.ratings = self.data["AvgResponseRating"].tolist()

        self.labels = self.data["Class"].tolist()
        self.classes = sorted(self.data["Class"].unique())

        self._initialize()


    def _get_image_path_and_label(self, index: int) -> tuple[str, str]:
        """
        Get the image path and label for a given index.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        tuple[str, str]
            A tuple containing the image path and its corresponding label.

        """
        image_path = os.path.join(
            self.images_path,
            str(self.data.loc[index, 'Class']),
            str(self.data.loc[index, 'ImageID']) + str(self.data.loc[index, 'Extension']),
        )
        label = self.labels[index]

        return image_path, label
