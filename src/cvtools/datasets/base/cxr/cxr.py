"""
Dataloader for NIH Chest X-Ray dataset: https://nihcc.app.box.com/v/ChestXray-NIHCC

Author: Atif Khurshid
Created: 2025-05-20
Modified: 2025-05-23
Version: 2.1

Changelog:
    - 2025-05-23: Add translation between class names and labels
    - 2025-05-22: Remove pytorch dependency and refactor code
    - 2025-05-22: Add image_size parameter for resizing images
"""
import os
import numpy as np
import pandas as pd

from ....image import imread


class CXRDataset():
    def __init__(
            self, root_dir, image_size=None, train=True, binary=True):
        """
        Parameters
        ----------
        root_dir : str
            Path to the root directory of the dataset.
        image_size : tuple, optional
            Size of the images to be resized to (height, width). Default is None.
        train : bool, optional
            If True, load training/validation data. If False, load test data. Default is True.
        binary : bool, optional
            If True, convert labels to binary (0 for 'No Finding', 1 for 'Finding'). Default is True.
        """

        self.root_dir = root_dir
        self.image_size = image_size    # (height, width)

        self.images_dir = os.path.join(self.root_dir, 'images')
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"Directory {self.images_dir} does not exist.")
        
        # Load annotations file
        self.data = pd.read_csv(os.path.join(self.root_dir, 'Data_Entry_2017_v2020.csv'))

        if train:
            # Read list of train/val indices
            with open(os.path.join(self.root_dir, 'train_val_list.txt'), 'r') as f:
                train_val_list = f.read().split('\n')
            # Filter the data to include only the train/val indices
            self.data = self.data[self.data['Image Index'].isin(train_val_list)]
        else:
            # Read list of test indices
            with open(os.path.join(self.root_dir, 'test_list.txt'), 'r') as f:
                test_list = f.read().split('\n')
            # Filter the data to include only the test indices
            self.data = self.data[self.data['Image Index'].isin(test_list)]
        # Reset the index of the DataFrame to ensure it is sequential
        self.data = self.data.reset_index(drop=True)
        
        # Convert the multiclass textual labels to binary - 0 for 'No Finding' and 1 for 'Finding'
        if binary:
            self.data['Finding Labels'] = self.data['Finding Labels'].apply(lambda x: "Normal" if x == 'No Finding' else "Abnormal")

        self.classes = self.data['Finding Labels'].unique().tolist()
        self.label2idx = {x : i for i, x in enumerate(self.classes)}
        self.idx2label = {v : k for k, v in self.label2idx.items()}


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        # Read filename from the DataFrame to complete image path
        img_path = os.path.join(self.images_dir, self.data.loc[idx, 'Image Index'])
        # Read image as grayscale
        image = imread(img_path, mode="L", size=self.image_size)
        # Add channel dimension
        image = image[..., np.newaxis]
        # Read label from the DataFrame and convert to index
        label = self.label2idx[self.data.loc[idx, 'Finding Labels']]

        return image, label
