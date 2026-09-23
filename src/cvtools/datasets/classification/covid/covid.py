"""
Dataloader for SIIM-FISABIO-RSNA COVID-19 Detection Challenge dataset: https://www.kaggle.com/competitions/siim-covid19-detection/
"""

# Author: Atif Khurshid
# Created: 2026-09-23
# Modified: None
# Version: 1.0
# Changelog:
#     - 2026-09-23: Initial version.

import os
import glob
from typing import Optional, Union

import numpy as np
import pandas as pd

from .._base import _ClassificationBaseImage


class Covid19Dataset(_ClassificationBaseImage):

    def __init__(
        self,
        root_dir: str,
        class_mode: str = "multiclass",
        annotation_level: str = "study",
        study_level_mode: str = "first",
        image_mode: str = "GRAY",
        image_scale: Optional[float] = None,
        image_size: Optional[Union[int, tuple[int, int]]] = None,
        preserve_aspect_ratio: bool = True,
        interpolation: Optional[int] = None,
    ):
        """
        SIIM-FISABIO-RSNA COVID-19 Detection Challenge dataset loader.

        The dataset is expected to be organized in the following structure:
        - root_dir/
            - _image_level.csv
            - _study_level.csv
            - 0a1a3dd9e738/
                - 79de130ea278/
                    - 64a776818efe.jpg
            - ...
        The root folder contains a folder for each study, and each study folder contains
        a folder for each series, which in turn contains the image files in jpg. This is
        the same structure as the original dataset, but with images converted to jpg format.

        Parameters
        ----------
        root_dir : str
            Path to the root directory of the dataset.
        class_mode : str, optional
            Mode for class labels. Can be "binary" (normal vs abnormal) or "multiclass" (original).
        image_mode : str, optional
            Mode to read images. Default is "GRAY" for grayscale images.
        annotation_level : str, optional
            Level of annotation to use. Can be "study" or "image". Default is "study".
        study_level_mode : str, optional
            Mode for study-level annotations. Can be "first" (use the first image of each study)
            or "random" (randomly select an image from each study). Default is "first".
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

        # Read annotations file
        self.data = pd.read_csv(os.path.join(self.root_dir, '_study_level.csv'))
        self.data["id"] = self.data["id"].str.removesuffix("_study")

        # Align the DataFrame with the actual study folders present in the root directory
        study_folders = [name for name in os.listdir(self.root_dir) \
                         if os.path.isdir(os.path.join(self.root_dir, name))]
        self.data = self.data[self.data["id"].isin(study_folders)].reset_index(drop=True)

        # Convert from one-hot encoded columns to a single label column
        class_cols = [
            'Negative for Pneumonia',
            'Typical Appearance',
            'Indeterminate Appearance',
            'Atypical Appearance'
        ]
        self.data['labels'] = self.data[class_cols].idxmax(axis=1)
        self.data = self.data.drop(columns=class_cols).reset_index(drop=True)

        # If class_mode is binary, convert the labels to 'Normal' and 'Abnormal'
        if class_mode == "binary":
            self.data['labels'] = np.where(self.data['labels'] == 'Negative for Pneumonia', 'Normal', 'Abnormal')

        # Each study can contain multiple images, so we need to explode the DataFrame to have one row per image
        def get_all_pngs(id_name, root_dir):
            id_dir = os.path.join(root_dir, id_name)
            pattern = os.path.join(id_dir, '**', '*.[pP][nN][gG]')
            paths = sorted(glob.glob(pattern, recursive=True))
            return [os.path.relpath(path, root_dir) for path in paths]
        
        self.data['image_path'] = self.data['id'].apply(lambda x: get_all_pngs(x, self.root_dir))
        self.data = self.data.explode('image_path').reset_index(drop=True)

        assert annotation_level in ["study", "image"], \
            f"Invalid annotation_level: {annotation_level}. Must be 'study' or 'image'."
        # In study mode, we need to pick one image per study.
        # We can either pick the first image or a random image from each study.
        if annotation_level == "study":
            if study_level_mode == "first":
                self.data = self.data.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)
            elif study_level_mode == "random":
                self.data = self.data.groupby('id').apply(
                    lambda x: x.sample(1)).reset_index(drop=True)
            else:
                raise ValueError(f"Invalid study_level_mode: {study_level_mode}. Must be 'first' or 'random'.")

        self.labels = self.data['labels'].tolist()
        self.classes = sorted(self.data['labels'].unique().tolist())

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
            self.root_dir,
            str(self.data.loc[index, 'image_path'])
        )
        label = self.labels[index]

        return image_path, label
