"""
Dataloader for a subset of the NIH Chest X-Ray dataset with pneumonia classification.
"""

# Author: Atif Khurshid
# Created: 2026-08-04
# Modified: None
# Version: 1.0
# Changelog:
#     - 2026-08-04: Initial version.

import os
import json
from typing import Optional, Union

import numpy as np
import pandas as pd

from .._base import _ClassificationBaseImageHDF5


class CXRPneumoniaDataset(_ClassificationBaseImageHDF5):

    def __init__(
        self,
        root_dir: str,
        class_mode: str = "original",
        view: str = "both",
        hdf5_mode: Optional[str] = None,
        image_mode: str = "GRAY",
        image_scale: Optional[float] = None,
        image_size: Optional[Union[int, tuple[int, int]]] = None,
        preserve_aspect_ratio: bool = True,
        interpolation: Optional[int] = None,
    ):
        """
        NIH Chest X-Ray pneumonia dataset loader.

        Reference:
            Shih et al. Radiology: Artificial Intelligence. (2020).
            Augmenting the National Institutes of Health Chest Radiograph Dataset
            with Expert Annotations of Possible Pneumonia

        This class loads a subset of images and labels from the NIH Chest X-Ray dataset.
        Follow the instructions for the CXRDataset class, and additionally download
            - stage_2_detailed_class_info.csv from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
            - pneumoia-challenge-dataset-mappings_2018.json from https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pneumonia-detection-challenge-2018
        and place them in pneumonia_challenge/ folder in the root_dir of the dataset.

        Parameters
        ----------
        root_dir : str
            Path to the root directory of the dataset.
        class_mode : str, optional
            Mode of classification. Can be "original" (three classes) or "binary" (normal vs abnormal).
            Default is "original".
        view : str, optional
            View position of the chest X-ray images to load.
            Can be "AP" (Anterior-Posterior), "PA" (Posterior-Anterior), or both.
            Default is "AP".
        hdf5_mode : str, optional
            If specified, load images from the given HDF5 file instead of from images folder.
             Default is None (load from images).
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
            hdf5_mode=hdf5_mode,
            image_mode=image_mode,
            image_scale=image_scale,
            image_size=image_size,
            preserve_aspect_ratio=preserve_aspect_ratio,
            interpolation=interpolation
        )

        if hdf5_mode:
            self.images_dir = ""

        else:
            self.images_dir = os.path.join(self.root_dir, 'images')
            if not os.path.exists(self.images_dir):
                raise FileNotFoundError(f"Directory {self.images_dir} does not exist.")

        # Load annotations file
        self.data = pd.read_csv(os.path.join(self.root_dir, 'Data_Entry_2017_v2020.csv'))

        mapping_path = os.path.join(
            root_dir,
            "pneumonia_challenge",
            "pneumonia-challenge-dataset-mappings_2018.json"
        )
        with open(mapping_path, "r") as f:
            mapping_data = json.load(f)
        mapping_df = pd.DataFrame(mapping_data)
        mapping_df = mapping_df.rename(columns={
            'img_id': 'Image Index',
            'subset_img_id': 'patientId'
        })

        labels_path = os.path.join(
            root_dir,
            "pneumonia_challenge",
            "stage_2_detailed_class_info.csv"
        )
        labels_df = pd.read_csv(labels_path)
        labels_df = labels_df.drop_duplicates(subset="patientId", keep="first")

        labels_df = labels_df.merge(mapping_df, on="patientId", how="inner")

        self.data = self.data.merge(labels_df, on="Image Index", how="inner")

        # Filter data based on view position
        assert view in ["AP", "PA", "both"], \
            f"Invalid view position: {view}. Must be 'AP', 'PA', or 'both'."
        if view == "AP":
            self.data = self.data[self.data["View Position"] == "AP"]
        elif view == "PA":
            self.data = self.data[self.data["View Position"] == "PA"]

        assert class_mode in ["original", "binary"], \
            f"Invalid class_mode: {class_mode}. Must be 'original' or 'binary'."
        if class_mode == "binary":
            self.data['class'] = self.data['class'].apply(
                lambda x: 'Abnormal' if x != "Normal" else "Normal"
            )

        self.labels = self.data['class'].tolist()
        self.classes = sorted(self.data['class'].unique().tolist())

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
            self.images_dir,
            str(self.data.loc[index, 'Image Index'])
        )
        label = self.labels[index]

        return image_path, label
