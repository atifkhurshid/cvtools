"""
PyTorch Classification Dataset Configuration Class.
"""

# Author: Atif Khurshid
# Created: 2026-05-08
# Modified: 2026-10-22
# Version: 1.3
# Changelog:
#     - 2026-05-08: Initial version.
#     - 2026-05-15: Refactored according to updated base class.
#     - 2026-05-15: Removed datasets from configuration.
#     - 2026-05-19: Allowed for optional validation and test datasets.
#     - 2026-10-22:

from typing import Optional, Union

import torch
import numpy as np

from torchvision.transforms.v2 import Compose
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight

from ..._base import ConfigBase
from ....datasets.classification.pytorch import PyTorchDataset
from .transforms import TransformSpec, build_transforms
from ....utils.pytorch import dataset_train_test_split


class PyTorchClassificationDatasetConfig(ConfigBase):

    def __init__(self, **kwargs):
        """
        This class provides a single-call interface to load datasets and create dataloaders
        for PyTorch classification tasks.

        Simply set the dataset class, dataset kwargs, and other configuration parameters,
        and call `load_data()` to get the train, validation, and test dataloaders.

        Parameters
        ----------
        dataset_name: str
            The name of the dataset (e.g., "CIFAR10", "MNIST").
        dataset_class: type[Dataset]
            The PyTorch dataset class to use (e.g., torchvision.datasets.CIFAR10).
        dataset_kwargs: dict
            Keyword arguments to pass to the dataset class constructor.
        trainset_kwargs: dict
            Additional keyword arguments to pass only to the training dataset constructor.
            These include parameters like `train=True` for creating separate train/test splits.
        valset_kwargs: dict
            Additional keyword arguments to pass only to the validation dataset constructor.
            Similar to `trainset_kwargs`, but for validation data.
        testset_kwargs: dict
            Additional keyword arguments to pass only to the test dataset constructor.
            Similar to `trainset_kwargs`, but for test data.
        class_mode: str
            The type of classification task. Options are "multilabel", "multiclass", or "binary".
        color_mode: str
            The color mode of the images. Options are "original", "RGB" or "GRAY".
        size_mode: str
            The size mode of the images. Options are "original", "fixed" or "variable".
        allow_variable_size: bool
            Whether to allow variable image sizes.
            If False, images will be resized to a fixed size even if size_mode is "variable".
        grayscale_channels: int
            The number of channels for grayscale images. Default is 1.
        fixed_image_size: Optional[tuple]
            The fixed image size to resize to when size_mode is "fixed" or when allow_variable_size is False.
        min_side_length: int
            The minimum side length for resizing when size_mode is "variable" and allow_variable_size is False.
        augmentations: bool
            Whether to apply data augmentations during training.
        test_size: float
            The proportion of the dataset to include in the test split (between 0.0 and 1.0).
            This is used only if `testset_kwargs` is empty, meaning no explicit test dataset is provided.
        val_size: float
            The proportion of the dataset to include in the validation split (between 0.0 and 1.0).
            This is used only if `valset_kwargs` is empty, meaning no explicit validation dataset is provided.
        random_state: int
            Random seed for reproducibility when splitting datasets.
        batch_size: int
            The batch size for the dataloaders.
        num_workers: int
            The number of subprocesses to use for data loading.
        pin_memory: bool
            Whether to pin memory in the dataloaders.
        persistent_workers: bool
            Whether to keep data loader workers alive between epochs.
        image_transform_specs_train: list[TransformSpec]
            A list of TransformSpec objects defining the image transformations to apply during training.
        image_transform_specs_test: list[TransformSpec]
            A list of TransformSpec objects defining the image transformations to apply during testing/validation.
        target_transform_specs: list[TransformSpec]
            A list of TransformSpec objects defining the target transformations to apply.

        Attributes
        ----------
        classes: list
            The list of class labels in the dataset.
        n_classes: int
            The number of classes in the dataset.
        class_weights: torch.Tensor
            The computed class weights for handling imbalanced datasets.
        train_batches: int
            The number of batches in the training dataloader.
        val_batches: int
            The number of batches in the validation dataloader.
        test_batches: int
            The number of batches in the test dataloader.
        train_samples: int
            The number of samples in the training dataset.
        val_samples: int
            The number of samples in the validation dataset.
        test_samples: int
            The number of samples in the test dataset.
        """
        super().__init__()

        self.dataset_name = ""
        self.dataset_class: Optional[type[Dataset]] = None
        self.dataset_kwargs: dict = {}
        self.trainset_kwargs: dict = {}
        self.valset_kwargs: dict = {}
        self.testset_kwargs: dict = {}

        self.class_mode: str = "multiclass"
        self.color_mode: str = "original"
        self.size_mode: str = "original"
        self.allow_variable_size: bool = False
        self.grayscale_channels: int = 1
        self.fixed_image_size: Optional[tuple] = (224, 224)
        self.min_side_length: int = 256
        self.augmentations: bool = False

        self.test_size: float = 0.2
        self.val_size: float = 0.1
        self.random_state: int = 42

        self.batch_size: int = 64
        self.num_workers: int = 4
        self.pin_memory: bool = True
        self.persistent_workers: bool = False

        self.image_transform_specs_train: list[TransformSpec] = []
        self.image_transform_specs_test: list[TransformSpec] = []
        self.target_transform_specs: list[TransformSpec] = []

        if type(self) is PyTorchClassificationDatasetConfig:
            # Only run if this is the instantiated class,
            # and not a parent class being inherited from.
            self._update_from_kwargs(**kwargs)

        default_transform_specs = self._get_default_transform_specs()
        if self.image_transform_specs_train == []:
            self.image_transform_specs_train = default_transform_specs[0]
        if self.image_transform_specs_test == []:
            self.image_transform_specs_test = default_transform_specs[1]
        if self.target_transform_specs == []:
            self.target_transform_specs = default_transform_specs[2]

        self.classes: list
        self.n_classes: int
        self.class_weights: torch.Tensor

        self.train_batches: int
        self.val_batches: int
        self.test_batches: int
        self.train_samples: int
        self.val_samples: int
        self.test_samples: int


    def load_data(
            self,
        ) -> tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        """
        Load datasets and compute class weights.

        Returns
        -------
        dataset_train: DataLoader
            Training dataset.
        dataset_val: Optional[DataLoader]
            Validation dataset.
        dataset_test: Optional[DataLoader]
            Test dataset.
        """
        dataset_train = self.dataset_class(
            **self.dataset_kwargs,
            **self.trainset_kwargs,
        )

        if hasattr(dataset_train, "labels"):
            train_labels = dataset_train.labels
        elif hasattr(dataset_train, "targets"):
            train_labels = dataset_train.targets
        elif hasattr(dataset_train, "_samples"):
            train_labels = np.array([y for _, y in dataset_train._samples])
        else:
            train_labels = np.array([y for _, y in dataset_train])

        if (
            isinstance(train_labels, np.ndarray) and
            train_labels.ndim > 1 and
            not self.class_mode == "multilabel"
        ):
            train_labels = train_labels.flatten().tolist()

        if len(self.testset_kwargs) > 0:
            dataset_test = self.dataset_class(
                **self.dataset_kwargs,
                **self.testset_kwargs,
            )
        elif self.test_size > 0.0:
            dataset_train, dataset_test = dataset_train_test_split(
                dataset_train,
                train_labels,
                test_size = self.test_size,
                random_state = self.random_state,
                multilabel = self.class_mode == "multilabel",
            )
            train_labels = np.array(train_labels)[dataset_train.indices]
        else:
            dataset_test = None

        if len(self.valset_kwargs) > 0:
            dataset_val = self.dataset_class(
                **self.dataset_kwargs,
                **self.valset_kwargs,
            )
        elif self.val_size > 0.0:
            dataset_train, dataset_val = dataset_train_test_split(
                dataset_train,
                train_labels,
                test_size = self.val_size,
                random_state = self.random_state,
                multilabel = self.class_mode == "multilabel",
            )
        else:
            dataset_val = None

        image_transforms_train = build_transforms(self.image_transform_specs_train)
        image_transforms_test = build_transforms(self.image_transform_specs_test)
        target_transforms = build_transforms(self.target_transform_specs)

        if hasattr(dataset_train, "classes"):
            self.classes = dataset_train.classes
        elif hasattr(dataset_train, "info") and "label" in dataset_train.info:
            self.classes = list(dataset_train.info["label"].values())
        else:
            self.classes = sorted(list(set(train_labels)))

        self.n_classes = len(self.classes)
        self._compute_class_weights(train_labels)

        dataset_train = PyTorchDataset(
            dataset_train,
            image_transforms_train,
            target_transforms
        )
        dataloader_train = self._build_dataloader(dataset_train, train=True)
        self.train_samples = len(dataset_train)
        self.train_batches = len(dataloader_train)

        if dataset_val is not None:
            dataset_val = PyTorchDataset(
                dataset_val,
                image_transforms_test,
                target_transforms
            )
            dataloader_val = self._build_dataloader(dataset_val, train=False)
            self.val_samples = len(dataset_val)
            self.val_batches = len(dataloader_val)
        else:
            dataloader_val = None
            self.val_samples = 0
            self.val_batches = 0

        if dataset_test is not None:
            dataset_test = PyTorchDataset(
                dataset_test,
                image_transforms_test,
                target_transforms
            )
            dataloader_test = self._build_dataloader(dataset_test, train=False)
            self.test_samples = len(dataset_test)
            self.test_batches = len(dataloader_test)
        else:
            dataloader_test = None
            self.test_samples = 0
            self.test_batches = 0

        return dataloader_train, dataloader_val, dataloader_test


    def _get_default_transform_specs(
            self,
        ) -> tuple[list[TransformSpec], list[TransformSpec], list[TransformSpec]]:
        """
        Get default transform specifications for image and target transforms.

        Returns
        -------
        image_transform_specs_train: list[TransformSpec]
            A list of TransformSpec objects defining the image transformations to apply during training.
        image_transform_specs_test: list[TransformSpec]
            A list of TransformSpec objects defining the image transformations to apply during testing/validation.
        target_transform_specs: list[TransformSpec]
            A list of TransformSpec objects defining the target transformations to apply.
        """
        basic_image_transform_specs = [
            TransformSpec(name="to_image"),
        ]

        if self.color_mode == "RGB":
            basic_image_transform_specs.append(
                TransformSpec(name="rgb")
            )
        elif self.color_mode == "GRAY":
            basic_image_transform_specs.append(
                TransformSpec(
                    name="grayscale", kwargs={"num_output_channels": self.grayscale_channels}
                )
            )
        elif self.color_mode == "original":
            pass  # No color conversion needed for original mode
        else:
            raise ValueError(f"Unsupported color mode: {self.color_mode}. Supported modes are 'original', 'RGB' and 'GRAY'.")

        if self.augmentations:
            augmented_image_transform_specs = []
            if self.size_mode == "fixed" or not self.allow_variable_size:
                augmented_image_transform_specs.append(
                    TransformSpec(
                        name="random_resized_crop", kwargs={"size": self.fixed_image_size}
                    )
                )
            if self.color_mode == "RGB":
                augmented_image_transform_specs.append(
                    TransformSpec(name="random_grayscale")
                )
            augmented_image_transform_specs.extend([
                TransformSpec(name="random_horizontal_flip"),
                TransformSpec(
                    name = "color_jitter",
                    kwargs = {
                        "brightness": 0.5,
                        "contrast": 0.5,
                        "saturation": 0.5,
                        "hue": 0.25
                    }
                ),
            ])
        else:
            augmented_image_transform_specs = []

        unaugmented_image_transform_specs = []
        if self.size_mode == "fixed":
            unaugmented_image_transform_specs.append(
                TransformSpec(
                    name="resize", kwargs={"size": self.fixed_image_size}
                )
            )
        elif self.size_mode == "variable":
            unaugmented_image_transform_specs.append(
                TransformSpec(name="resize", kwargs={"size": self.min_side_length}),
            )
            if not self.allow_variable_size:
                unaugmented_image_transform_specs.append(
                    TransformSpec(name="center_crop", kwargs={"size": self.fixed_image_size}),
                )
        elif self.size_mode == "original":
            pass  # No resizing or cropping needed for original size
        else:
            raise ValueError(f"Unsupported size mode: {self.size_mode}. Supported modes are 'original', 'fixed' and 'variable'.")

        final_image_transform_specs = [
            TransformSpec(
                name="to_dtype",
                kwargs={"dtype": torch.float32, "scale": True}
            )
        ]

        if self.augmentations:
            image_transform_specs_train = (
                basic_image_transform_specs +
                augmented_image_transform_specs +
                final_image_transform_specs
            )
        else:
            image_transform_specs_train = (
                basic_image_transform_specs +
                unaugmented_image_transform_specs +
                final_image_transform_specs
            )

        image_transform_specs_test = (
            basic_image_transform_specs +
            unaugmented_image_transform_specs +
            final_image_transform_specs
        )

        target_transform_specs = [
            TransformSpec(name="to_int64_tensor")
        ]

        return image_transform_specs_train, image_transform_specs_test, target_transform_specs


    def _compute_class_weights(
            self,
            labels: Union[list, np.ndarray, torch.Tensor]
        ):
        """
        Compute class weights for imbalanced datasets.
        """
        if self.class_mode == "multilabel":
            pos_counts = torch.zeros(self.n_classes)

            for label in labels:
                for l in label:
                    i = self.classes.index(l)
                    pos_counts[i] += 1

            total = len(labels)
            neg_counts = total - pos_counts

            self.class_weights = neg_counts / pos_counts.clamp(min=1)  # clamp avoids division by zero

        elif self.class_mode == "multiclass" or self.class_mode == "binary":
            
            weights = compute_class_weight(
                'balanced', classes = np.unique(labels), y = labels
            )
            self.class_weights = torch.tensor(weights, dtype=torch.float32)

        else:
            raise ValueError(f"Unsupported class mode: {self.class_mode}. Supported modes are 'multilabel', 'multiclass', and 'binary'.")


    def _build_dataloader(
            self,
            dataset: Dataset,
            train: bool
        ) -> DataLoader:
        """
        Create dataloaders for train, val or test datasets.

        Parameters
        ----------
        dataset: Dataset
            The dataset for which to create the dataloader.
        train: bool
            Whether the dataloader is for training (True) or validation/testing (False).
        """
        if (
            self.size_mode == "original" or
            (self.size_mode == "variable" and self.allow_variable_size)
        ):
            if self.batch_size != 1:
                print("Warning: Batch size must be 1 for variable image sizes. Setting batch_size to 1.")
            self.batch_size = 1

        dataloader = DataLoader(
            dataset,
            shuffle = train,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            pin_memory = self.pin_memory,
            persistent_workers = self.persistent_workers,
        )

        return dataloader
