"""
Dataloader for RSNA Screening Mammography dataset.
"""

# Author: Atif Khurshid
# Created: 2026-10-15
# Modified: None
# Version: 1.0
# Changelog:
#     - 2026-10-15: Initial version.

import os
from typing import Optional, Union

import numpy as np
import pandas as pd

from .._base import _ClassificationBaseImage


class MammogramDataset(_ClassificationBaseImage):

    def __init__(
        self,
        root_dir: str,
        site: str = "both",
        level: str = "patient",
        view: str = "all",
        class_mode: str = "multiclass",
        split = "train",
        balanced_classes: bool = False,
        same_direction: bool = False,
        exclude_implants: bool = False,
        image_mode: str = "GRAY",
        image_scale: Optional[float] = None,
        image_size: Optional[Union[int, tuple[int, int]]] = None,
        preserve_aspect_ratio: bool = True,
        interpolation: Optional[int] = None,
    ):
        """
        Mammogram dataset from Trivedi et al., "Open-Source Dataset for the RSNA Screening Mammography Cancer Detection Challenge". 2026.
        Downloaded from https://www.kaggle.com/datasets/theoviel/rsna-breast-cancer-1024-pngs

        The dataloader requires all images to be in an "images" folder in the root_dir.
        The image files should be named as <patient_id>_<image_id>.png.
        Alongside the images should be the train.csv file containing the metadata and labels for all train images.

        Parameters
        ----------
        root_dir : str
            Path to the root directory of the dataset.
        site: str, optional
            The source of the mammogram images. Options are "both", "1", "2".
        level : str, optional
            The level of the dataset. Default is "patient". Other option is "image".
        view : str, optional
            The view of the mammogram images. Options are "all", "CC", "MLO".
        class_mode : str, optional
            The classification mode. Options are "multi-class", "cancer", and "abnormal".
                - "multiclass": Classifies images into 3 classes: normal, benign, and cancer.
                - "cancer": Classifies images into 2 classes: cancer and non-cancer.
                - "abnormal": Classifies images into 2 classes: normal and abnormal (benign + cancer).
        split : str, optional
            The dataset split to use. Options are "train" and "test".
        balanced_classes : bool, optional
            If True, the dataset will be balanced with respect to the classes.
        same_direction : bool, optional
            If True, the images will be preprocessed to have the same direction.
        exclude_implants : bool, optional
            If True, images with implants will be excluded from the dataset.
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
        """
        if level not in ["patient", "image"]:
            raise ValueError(f"Invalid level: {level}. "
                             f"Valid options are 'patient', 'image'.")

        if view not in ["all", "CC", "MLO"]:
            raise ValueError(f"Invalid view: {view}. "
                             f"Valid options are 'all', 'CC', 'MLO'.")

        if site not in ["both", "1", "2"]:
            raise ValueError(f"Invalid site: {site}. "
                             f"Valid options are 'both', '1', '2'.")

        super().__init__(
            root_dir=root_dir,
            image_mode=image_mode,
            image_scale=image_scale,
            image_size=image_size,
            preserve_aspect_ratio=preserve_aspect_ratio,
            interpolation=interpolation
        )
        self.root_dir = root_dir
        if same_direction:
            self.images_path = os.path.join(root_dir, "images_roi_flipped")
        else:
            self.images_path = os.path.join(root_dir, "images")

        if split == "train":
            self.data = pd.read_csv(os.path.join(self.root_dir, "train_split.csv"))
        elif split == "test":
            self.data = pd.read_csv(os.path.join(self.root_dir, "test_split.csv"))
        else:
            raise ValueError(f"Invalid split: {split}. "
                             f"Valid options are 'train', 'test'.")

        if view == "CC":
            self.data = self.data[self.data["view"] == "CC"].reset_index(drop=True)
        elif view == "MLO":
            self.data = self.data[self.data["view"] == "MLO"].reset_index(drop=True)

        if site == "1":
            self.data = self.data[self.data["site_id"] == 1].reset_index(drop=True)
        elif site == "2":
            self.data = self.data[self.data["site_id"] == 2].reset_index(drop=True)

        if exclude_implants:
            self.data = self.data[self.data["implant"] == 0].reset_index(drop=True)

        if class_mode == "multiclass":
            self.data["class"] = self.data.apply(
                lambda row: "Cancer" if row["cancer"] == 1 else ("Benign" if row["BIRADS"] == 2 else "Normal"),
                axis=1
            )
        elif class_mode == "cancer":
            self.data["class"] = self.data["cancer"].apply(
                lambda x: "Cancer" if x == 1 else "Non-Cancer"
            )
        elif class_mode == "abnormal":
            self.data["class"] = self.data.apply(
                lambda row: "Abnormal" if row["cancer"] == 1 or row["BIRADS"] == 2 else "Normal",
                axis=1
            )
        else:
            raise ValueError(f"Invalid class_mode: {class_mode}. "
                             f"Valid options are 'multiclass', 'cancer', 'abnormal'.")

        if balanced_classes:
            rng = np.random.default_rng(42)

            severity = {
                "Cancer": 2,
                "Abnormal": 1,
                "Benign": 1,
                "Normal": 0,
                "Non-Cancer": 0,
            }

            sev = self.data["class"].map(severity)
            idx = sev.groupby(self.data["patient_id"]).idxmax()
            patient_label = self.data.loc[idx].set_index("patient_id")["class"]

            n = patient_label.value_counts().min()
            keep_ids = np.concatenate([
                rng.choice(
                    patient_label[patient_label == c].index.to_numpy(),
                    size=n,
                    replace=False,
                )
                for c in patient_label.unique()
            ])

            self.data = self.data[self.data["patient_id"].isin(keep_ids)].reset_index(drop=True)

        if level == "patient":
            self.data = (
                self.data.sort_values("cancer", ascending=False, kind="stable")
                    .drop_duplicates(subset="patient_id", keep="first")
                    .sort_index()
                    .reset_index(drop=True)
            )

        self.labels = self.data["class"].tolist()
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
            str(self.data.loc[index, 'patient_id']) + "_" +
            str(self.data.loc[index, 'image_id']) + ".png",
        )
        label = self.labels[index]

        return image_path, label
