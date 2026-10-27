"""
Utility functions for PyTorch datasets.
"""

# Author: Atif Khurshid
# Created: 2025-09-08
# Modified: 2026-06-08
# Version: 1.3
# Changelog:
#     - 2025-09-22: Added InfiniteDataLoader function.
#     - 2026-02-19: Modified train_test_split to use sklearn function.
#     - 2026-05-08: Added train_samples_per_class option to train_test_split.

from typing import Optional, Union

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset as Subset_
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


class Subset(Subset_):

    def __init__(self, dataset: Dataset, indices: list[int]):
        super().__init__(dataset, indices)
    

    def __getattr__(self, name: str):
        # Only called when normal lookup fails.
        return getattr(self.dataset, name)


def dataset_train_test_split(
        dataset: Dataset,
        labels: Union[list, np.ndarray],
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
        train_samples_per_class: Optional[int] = None,
        random_state: Optional[int] = None,
        shuffle: bool = True,
        stratify: bool = True,
        multilabel: bool = False,
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
    train_samples_per_class : int, optional
        If specified, overrides train_size and ensures that the training set
        contains exactly this many samples per class.
    random_state : int, optional
        Controls the randomness of the split.
        Pass an int for reproducible output across multiple function calls.
    shuffle : bool, optional
        Whether or not to shuffle the data before splitting. Default is True.
        Shuffle must be True if stratify is True.
    stratify : bool, optional
        Whether or not to perform stratified splitting. Default is True.
        If True, the data is split in a stratified fashion, using the labels provided.
    multilabel : bool, optional
        Whether the labels are multilabel (list of lists) or single-label (list of scalars).

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
    if multilabel:
        labels = MultiLabelBinarizer().fit_transform(labels)

    if train_samples_per_class is not None:
        rng = np.random.default_rng(random_state)
        if multilabel:
            selector_fn = _greedy_per_class_selection_multilabel
        else:
            selector_fn = _greedy_per_class_selection

        train_indices, test_indices = selector_fn(
            labels, train_samples_per_class, rng
        )

    elif multilabel and stratify:
        splitter = MultilabelStratifiedShuffleSplit(
            n_splits = 1,
            test_size = test_size,
            train_size = train_size,
            random_state = random_state,
        )
        train_indices, test_indices = next(
            splitter.split(np.zeros(len(labels)), labels)
        )

    else:
        train_indices, test_indices = train_test_split(
            np.arange(len(labels)),
            test_size = test_size,
            train_size = train_size,
            random_state = random_state,
            shuffle = shuffle,
            stratify = labels if stratify else None,
        )

    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def _greedy_per_class_selection(
        labels: np.ndarray,
        k: int,
        rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Select training indices so that each class has at least k samples.
    Classes are processed from rarest to most frequent. Among candidates,
    the sample causing the least overshoot on already satisfied classes is chosen.

    Parameters
    ----------
    labels : np.ndarray
        A 1D array of shape (n_samples,) indicating the class of each sample.
    k : int
        The minimum number of samples required for each class in the training set.
    rng : np.random.Generator
        A random number generator for reproducibility.
    """
    train_indices, test_indices = [], []
    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        if len(cls_indices) < k:
            raise ValueError(f"Class '{cls}' has only {len(cls_indices)} samples.")
        chosen = rng.choice(cls_indices, size=k, replace=False)
        train_indices.extend(chosen)
        test_indices.extend(np.setdiff1d(cls_indices, chosen))

    return np.array(train_indices), np.array(test_indices)


def _greedy_per_class_selection_multilabel(
        labels: np.ndarray,
        k: int,
        rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Select training indices so that each class has at least k positives.
    Classes are processed from rarest to most frequent. Among candidates,
    the sample causing the least overshoot on already satisfied classes is chosen.

    Parameters
    ----------
    labels : np.ndarray
        A binary matrix of shape (n_samples, n_classes) indicating the presence of each class in each sample.
    k : int
        The minimum number of positive samples required for each class in the training set.
    rng : np.random.Generator
        A random number generator for reproducibility.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing the training indices and testing indices.
    """
    n_samples, n_classes = labels.shape
    counts = np.zeros(n_classes, dtype=int)
    selected = np.zeros(n_samples, dtype=bool)

    for cls in np.argsort(labels.sum(axis=0)):
        while counts[cls] < k:
            cand = np.where((labels[:, cls] == 1) & ~selected)[0]
            if len(cand) == 0:
                raise ValueError(
                    f"Class '{cls}' cannot reach {k} training samples "
                    f"(reached {counts[cls]})."
                )
            overshoot = (labels[cand] * (counts >= k)).sum(axis=1)
            best = cand[overshoot == overshoot.min()]
            pick = rng.choice(best)
            selected[pick] = True
            counts += labels[pick]

    train_indices = np.where(selected)[0]
    test_indices = np.where(~selected)[0]

    return train_indices, test_indices


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
