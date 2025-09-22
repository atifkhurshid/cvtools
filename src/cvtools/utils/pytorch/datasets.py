"""
Utility functions for PyTorch datasets.
"""

# Author: Atif Khurshid
# Created: 2025-09-08
# Modified: 2025-09-22
# Version: 1.1
# Changelog:
#     - 2025-09-22: Added InfiniteDataLoader function.

import random
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset as Subset_


class Subset(Subset_):
    
    def __init__(self, dataset: Dataset, indices: list[int]):
        super().__init__(dataset, indices)
    

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)
    

def stratified_train_test_split(
        dataset: Dataset,
        labels: list,
        n_train_per_class: int
    ) -> tuple[Subset, Subset]:
    """
    Splits a dataset into stratified training and testing subsets.

    Parameters
    ----------
    dataset : Dataset
        The PyTorch dataset to split.
    labels : list
        List of labels corresponding to each sample in the dataset.
    n_train_per_class : int
        Number of training samples per class.

    Returns
    -------
    tuple[Subset, Subset]
        A tuple containing the training and testing subsets.
    
    Examples
    --------
    >>> from torchvision.datasets import MNIST
    >>> from torchvision import transforms
    >>> from cvtools.utils.pytorch import stratified_train_test_split
    >>> mnist_dataset = MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    >>> labels = mnist_dataset.targets.tolist()
    >>> train_subset, test_subset = stratified_train_test_split(mnist_dataset, labels, n_train_per_class=1000)
    >>> len(train_subset)
    10000
    >>> len(test_subset)
    50000
    """
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    train_indices, test_indices = [], []
    for label, indices in class_indices.items():
        random.shuffle(indices)
        train_indices.extend(indices[:n_train_per_class])
        test_indices.extend(indices[n_train_per_class:])

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
