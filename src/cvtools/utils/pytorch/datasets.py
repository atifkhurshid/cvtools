"""
Utility functions for PyTorch datasets.
"""

# Author: Atif Khurshid
# Created: 2025-09-08
# Modified: 2026-02-19
# Version: 1.2
# Changelog:
#     - 2025-09-22: Added InfiniteDataLoader function.
#     - 2026-02-19: Modified train_test_split to use sklearn function.

from typing import Optional, Union

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset as Subset_
from sklearn.model_selection import train_test_split


class Subset(Subset_):
    
    def __init__(self, dataset: Dataset, indices: list[int]):
        super().__init__(dataset, indices)
    

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)
    

def dataset_train_test_split(
        dataset: Dataset,
        labels: list,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        random_state: Optional[int] = None,
        shuffle: bool = True,
        stratify: bool = True,
    ) -> tuple[Subset, Subset]:
    """
    Splits a dataset into stratified training and testing subsets.

    Parameters
    ----------
    dataset : Dataset
        The PyTorch dataset to split.
    labels : list
        List of labels corresponding to each sample in the dataset.
    test_size : float or int, optional
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split.
        If int, represents the absolute number of test samples.
        If None, the value is set to the complement of the train size.
    train_size : float or int, optional
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split.
        If int, represents the absolute number of train samples.
        If None, the value is set to the complement of the test size.
    random_state : int, optional
        Controls the randomness of the split.
        Pass an int for reproducible output across multiple function calls.
    shuffle : bool, optional
        Whether or not to shuffle the data before splitting. Default is True.
        Shuffle must be True if stratify is True.
    stratify : bool, optional
        Whether or not to perform stratified splitting. Default is True.
        If True, the data is split in a stratified fashion, using the labels provided.

    Returns
    -------
    tuple[Subset, Subset]
        A tuple containing the training and testing subsets.
    
    Examples
    --------
    >>> from torchvision.datasets import MNIST
    >>> from torchvision import transforms
    >>> from cvtools.utils.pytorch import dataset_train_test_split
    >>> mnist_dataset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    >>> labels = mnist_dataset.targets.tolist()
    >>> train_subset, test_subset = dataset_train_test_split(mnist_dataset, labels, train_size=0.8)
    """
    train_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        test_size = test_size,
        train_size = train_size,
        random_state = random_state,
        shuffle = shuffle,
        stratify = labels if stratify else None,
    )

    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, test_subset


def InfiniteDataLoader(dataloader: DataLoader):
    """
    Creates an infinite generator from a PyTorch DataLoader.

    Parameters
    ----------
    dataloader : DataLoader
        The PyTorch DataLoader to create an infinite generator from.

    Yields
    ------
    batch
        Batches of data from the DataLoader.

    Examples
    --------
    >>> import torch
    >>> from torch.utils.data import DataLoader, TensorDataset
    >>> from cvtools.utils.pytorch import infinite_dataloader
    >>> dataset = TensorDataset(torch.arange(10).float().unsqueeze(1))
    >>> dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
    >>> inf_loader = InfiniteDataLoader(dataloader)
    >>> for _ in range(5):
    ...     batch = next(inf_loader)
    ...     print(batch)
    tensor([[2.],
            [0.],
            [1.]])
    tensor([[5.],
            [4.],
            [3.]])
    tensor([[8.],
            [6.],
            [7.]])
    tensor([[9.]])
    tensor([[2.],
            [1.],
            [0.]])
    """
    while True:
        for batch in dataloader:
            yield batch
