"""
Dataloader for Mammogram dataset from https://github.com/gistmammocnn/MammogramGist-CNN
"""

# Author: Atif Khurshid
# Created: 2026-09-24
# Modified: None
# Version: 1.1
# Changelog:
#     - 2026-09-24: Initial version.
#     - 2026-10-09: Added class modes.

import os
import glob
from typing import Optional, Union

import numpy as np
import pandas as pd

from .._base import _ClassificationBaseImage


class MammogramGistDataset(_ClassificationBaseImage):

    _class_modes = ["original", "binary-cancer", "binary-normal", "binary-exclude"]

    def __init__(
        self,
        root_dir: str,
        preprocessing: str = "none",
        class_mode: str = "original",
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
        class_mode : str, optional
            The mode of the classification task. Default is "original", which uses three classes.
            If set to "binary-cancer", the "Contralateral" class is merged with the "Cancer" class.
            If set to "binary-normal", the "Contralateral" class is merged with the "Normal" class.
            If set to "binary-exclude", the "Contralateral" class is excluded from the dataset.
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
        if class_mode not in self._class_modes:
            raise ValueError(f"Invalid class_mode: {class_mode}. "
                             f"Valid options are ", f"{', '.join(self._class_modes)}.")

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

        if class_mode == "binary-exclude":
            self.data = self.data[self.data["Class"] != "Contralateral"].reset_index(drop=True)

        no_human_annotations = self.data["AvgResponseRating"].isna()
        if train:
            self.data = self.data[no_human_annotations].reset_index(drop=True)
            self.ratings = None
        else:
            # Use only images with human annotations for testing
            self.data = self.data[~no_human_annotations].reset_index(drop=True)
            self.ratings = self.data["AvgResponseRating"].tolist()

        self.labels = self.data["Class"].tolist()

        if class_mode == "binary-cancer":
            self.labels = ["Cancer" if label == "Contralateral" else label for label in self.labels]
        elif class_mode == "binary-normal":
            self.labels = ["Normal" if label == "Contralateral" else label for label in self.labels]

        self.classes = sorted(set(self.labels))

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
