"""
Dataloader for Princeton SUN dataset: https://vision.princeton.edu/projects/2010/SUN/

Author: Atif Khurshid
Created: 2025-05-23
Modified: None
Version: 1.0

Changelog:
    - None
"""
import os
import numpy as np
import pandas as pd

from ....image import imread


class SUNDataset():
    def __init__(
            self,
            root_dir,
            class_hierarchy = "basic",
            image_size = None,
            preserve_aspect_ratio = True,
            train = True
        ):
        """
        Parameters
        ----------
        root_dir : str
            Path to the root directory of the dataset.
        class_hierarchy : str, optional
            Class hierarchy to use. Options are "sun", "basic", or "superordinate". Default is "basic".
        image_size : tuple, optional
            Size of the images to be resized to (height, width). Default is None.
        train : bool, optional
            If True, load training/validation data. If False, load test data. Default is True.
        """

        self.root_dir = root_dir
        self.image_size = image_size    # (height, width)
        self.preserve_aspect_ratio = preserve_aspect_ratio

        self.images_dir = os.path.join(self.root_dir, 'images')
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Directory {self.images_dir} does not exist.")
    
        if train:
            filepaths_list_path = os.path.join(self.root_dir, 'metadata', 'Training_01.txt')
        else:
            filepaths_list_path = os.path.join(self.root_dir, 'metadata', 'Testing_01.txt')

        # Read list of train images
        with open(filepaths_list_path, 'r') as f:
            filepaths_list = f.read().split('\n')[:-1]    # Remove last empty row

        # Infer class from filepath
        # Example: /a/airport/entrance/abckjaskasd.jpg
        # Class is the 2nd (or possibly including 3rd part) of the filepath
        self.filepaths = []
        self.labels = []
        for filepath in filepaths_list:
            parts = filepath.split("/")
            if len(parts) == 4:
                self.labels.append(parts[2])
            elif len(parts) == 5:
                self.labels.append(f"{parts[2]}/{parts[3]}")
            else:
                print(f"WARNING: Filepath {filepath} will be skipped because class could not be inferred.")
                continue
            self.filepaths.append(filepath[1:])    # Remove leading '/'

        assert len(self.filepaths) == len(self.labels), "Number of filepaths and classes do not match."

        if class_hierarchy != "sun":
            # Read class hierarchy from CSV file
            class_hierarchy_df = pd.read_csv(os.path.join(self.root_dir, 'metadata', 'class_hierarchy.csv'))
            class_hierarchy_df = class_hierarchy_df.set_index('class')
            # Change label name according to the class hierarchy
            self.labels = [class_hierarchy_df.loc[x, class_hierarchy] for x in self.labels]

        self.classes = list(set(self.labels))
        self.label2index = {x : i for i, x in enumerate(self.classes)}
        self.index2label = {v : k for k, v in self.label2index.items()}


    def __len__(self):
        return len(self.filepaths)


    def __getitem__(self, idx):
        # Read filepath from the list
        img_path = os.path.join(self.images_dir, self.filepaths[idx])
        # Read image as RGB
        image = imread(
            img_path,
            mode = "RGB",
            size = self.image_size,
            preserve_aspect_ratio = self.preserve_aspect_ratio,
        )
        # Read label from the list and convert to label index
        label = self.label2index[self.labels[idx]]

        return image, label
