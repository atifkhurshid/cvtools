"""
Utility functions for PyTorch datasets.
"""

# Author: Atif Khurshid
# Created: 2025-09-08
# Modified: None
# Version: 1.0
# Changelog:
#     - None

import random
from collections import defaultdict

from torch.utils.data import Dataset
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
