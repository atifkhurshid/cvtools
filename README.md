# Computer Vision Tools

A Python toolkit for computer vision research and prototyping. It provides image I/O and processing helpers, classification dataset loaders, PyTorch models and layers, loss functions, a configuration-driven training pipeline, evaluation metrics, and visualization utilities.


## Installation

Clone the repository and install it in your Python environment in *editable* mode using:

```bash
git clone https://github.com/atifkhurshid/cvtools.git
cd cvtools
pip install -e .[all]           # with all optional dependencies
```

Alternatively, you may also install only core or some optional dependencies using:

```bash
pip install -e "."              # core dependencies only
pip install -e ".[torch]"       # adds torch, torchvision, torchaudio
pip install -e ".[tensorflow]"  # adds tensorflow
pip install -e ".[dicom]"       # adds pydicom
```

---

## Quick Start

```python
import torch
from torchvision.datasets import Imagenette

from cvtools.configs.pytorch import PyTorchClassificationDatasetConfig
from cvtools.configs.pytorch import PyTorchClassificationModelConfig
from cvtools.models.pytorch import StandardCNN
from cvtools.models.pytorch.utils import train_classification_model
from cvtools.models.pytorch.utils import test_classification_model
from cvtools.evaluation.classification import evaluate_classification
from cvtools.utils.pytorch import set_seed


device = torch.accelerator.current_accelerator() \
    if torch.accelerator.is_available() else torch.device("cpu")
print(f"Using device: {device}")

set_seed(42, device)

dataset_config = PyTorchClassificationDatasetConfig(
    dataset_name = "Imagenette",
    dataset_class = Imagenette,
    dataset_kwargs = dict(
        root = IMAGENETTE_PATH,
        size = "full",
        download = True,
    ),
    trainset_kwargs = dict(
        split = "train",
    ),
    testset_kwargs = dict(
        split = "val",
    ),
    size_mode = "variable",
    color_mode = "RGB",
    augmentations = True,
)

dataloader_train, dataloader_val, dataloader_test = dataset_config.load_data()

model_config = PyTorchClassificationModelConfig(
    model_name = "resnet18",
    model_class = StandardCNN,
    model_kwargs = dict(
        name = "resnet18",
        n_classes = dataset_config.n_classes,
        pretrained = True,
    ),
    n_classes = dataset_config.n_classes,
    input_shape = (1, 3, 224, 224),
    device = device,
    loss_fn = "cross_entropy",
    class_weights = dataset_config.class_weights,
    optimizer = "sgd",
    metric = "multiclass_accuracy",
    epochs = 50,
    learning_rate = 1e-2,
    momentum = 0.9,
    weight_decay = 1e-4,
    log_frequency = 5,
    scheduler = "plateau",
    scheduler_patience = 5,
    pretrain = True,
    pretrain_epochs = 2,
    early_stopping = True,
    early_stopping_patience = 10,
    early_stopping_restore_best_weights = True,
    checkpoint_dir = "./checkpoints",
    checkpoint_interval = 10,
)

model = model_config.prepare_model()

# Optional logging to weights and biases
run = wandb.init(
    # args
)
# else run = None

train_classification_model(
    model = model,
    train_dataloader = dataloader_train,
    val_dataloader = dataloader_val,
    val_strategy = model_config.val_strategy,
    epochs = model_config.epochs,
    early_stopping = model_config.early_stopping,
    patience = model_config.early_stopping_patience,
    restore_best_weights = model_config.early_stopping_restore_best_weights,
    run = run,
    log_interval = dataset_config.train_batches // model_config.log_frequency,
    checkpoint_dir = model_config.checkpoint_dir,
    checkpoint_interval = model_config.checkpoint_interval,
)

mean_loss, mean_metric, outputs, y_pred, y_true = test_classification_model(
    model = model,
    data = dataloader_test,
    return_outputs = True,
)

report = evaluate_classification(
    y_true, y_pred, outputs,
    class_names = dataset_config.classes,
    confusion = True,
    roc = True,
    return_report = True
)
```

---

## Modules

The following is the overall layout of the package. Refer to the docstrings in each module for parameters and examples.

```
cvtools/
├── image/            Image I/O and resizing
├── datasets/         Classification dataset loaders
├── decomposition/    Dimensionality reduction
├── layers/           Custom PyTorch layers
├── losses/           Custom PyTorch loss functions
├── models/           PyTorch model base classes and training utilities
├── configs/          Config classes and transform registry
├── metrics/          Classification and clustering metrics
├── evaluation/       High-level model evaluation
├── visualization/    Plots, feature projections, Grad-CAM
└── utils/            Array, sampling, seeding, early stopping
```

### `cvtools.image`

OpenCV-based image utilities. A PyTorch variant lives in `image/pytorch.py`.

- `imread`, `imwrite`, `imshow`: read and write with OpenCV, display with matplotlib.
- `imscale`: scale by a factor.
- `imresize`: resize to a target size, optionally preserving aspect ratio.
- `imresize_maximum`, `imresize_minimum`: resize so the longest or shortest side matches a limit.
- `IMAGE_EXTENSIONS`: list of supported file extensions.
- `image.pytorch.imread`, `image.pytorch.imresize`: tensor-based equivalents.

### `cvtools.datasets.classification`

Dataset loaders that return `(image: np.ndarray, label: int)` pairs. All accept common options for resizing (`image_scale`, `image_size`, `preserve_aspect_ratio`, `interpolation`).

**Generic loaders**

- `ClassificationDataset`: folder-per-class image dataset with optional HDF5 caching.
- `SavedFeaturesDataset`: dataset of pre-extracted feature maps.

**Benchmark and domain-specific loaders**

- Scenes: `Scenes8Dataset` (Oliva & Torralba), `Scenes15Dataset` (Lazebnik et al.), `SUNDataset` (with class hierarchy support).
- ImageNet: `ImageNetDataset` (with WordNet ID mapping), `ImageNet64Dataset`.
- Chest X-ray (NIH): `CXRDataset` (with `visualize_hierarchy`), `CXRAbnormalityDataset`, `CXRPneumoniaDataset`.
- Other medical: `Covid19Dataset` (SIIM-FISABIO-RSNA, with DICOM preprocessing), `MammogramDataset` (RSNA), `MammogramGistDataset` (Evans et al.).

**Framework wrappers**

- `PyTorchDataset`: wraps any loader above (or any other loader with similar interface) as a `torch.utils.data.Dataset`, applying `transform` and `target_transform` and supporting batched `__getitems__`.
- `ClassificationDatasetTF`: TensorFlow `Sequence` wrapper of `ClassificationDataset`.

### `cvtools.decomposition`

- `PCA`: incremental PCA with automatic component selection by cumulative explained variance.

### `cvtools.layers.pytorch`

Custom `nn.Module` layers.

- Normalization: `DivisiveFeatureNorm`, `GaussianDivisiveNorm`, `LearnableDivisiveNorm`, `WeightedDivisiveNorm`, `SpatialLayerNorm`, `L2Norm`.
- Pooling: `CenterDependentPool2D` (eccentricity-dependent, with `visualize_masks`).
- Input handling: `SquarePad2D`, `GrayscaleToRGB`.

### `cvtools.losses.pytorch`

- `ArcLayer`, `ArcLoss`: ArcFace angular margin loss.
- `CenterLoss`: center loss for feature compactness.
- `SimCLRLoss`: contrastive loss for self-supervised learning.

### `cvtools.models.pytorch`

Model abstractions that bundle architecture, loss, optimizer, scheduler, and metric.

- `PyTorchModel`: abstract base class. Provides `configure_training`, `train_step`, `test_step`, `compute_loss`, `compute_metric`, `scheduler_step`, learning-rate getters/setters, forward-hook registration, and multilabel support.
- `PyTorchSequentialModel`: model built from a sequence of registered layers.
- `PyTorchCNNModel`: CNN base with `freeze_backbone` and `unfreeze_backbone`.
- `StandardCNN`: standard CNN implementations (supports 'AlexNet' to 'ResNet').
- `PixelClassifier`: classification directly from pixel values.
- Utilities (`models.pytorch.utils`):
  - `init_weights`: Kaiming uniform initialization.
  - `train_classification_model`, `test_classification_model`: training and inference loops.
  - `run_trials`: repeated train/test runs.
  - `extract_features`, `save_feature_maps`: feature extraction.

### `cvtools.configs`

Declarative setup for PyTorch classification experiments.

- `PyTorchClassificationDatasetConfig`: single-call `load_data()` that builds train, validation, and test dataloaders from a dataset class, transforms, and split settings.
- `PyTorchClassificationModelConfig`: single-call model, loss, optimizer, scheduler, early stopping, and checkpoint settings, with `prepare_model()`.
- Transform registry (`configs.pytorch.datasets.transforms`):
  - `TransformSpec(name, kwargs)`: serializable transform description.
  - `register_transform(name)`: decorator to register custom transforms.
  - `build_transforms(specs)`: builds a `torchvision.transforms.v2.Compose`.
  - Preregistered: most `torchvision.transforms.v2` transforms (for example `"resize"`, `"random_crop"`, `"color_jitter"`, `"normalize"`).
  - Custom target transforms: `ToInt64Tensor`, `ToFloatTensor`, `ToOneHotTensor`, `SqueezeTensor`.

### `cvtools.metrics`

Implementations of different evaluation metrics.

- `classification`: `compute_roc`, `compute_multilabel_roc`, `multilabel_auc`, `multilabel_accuracy`.
- `clustering`: `silhouette_score`, `clustering_accuracy` (Kuhn-Munkres), `clustering_f_measure`, `clustering_purity`, `maximum_cluster_assignment_score`, `intra_cluster_variability`, `inter_cluster_variability`.

### `cvtools.evaluation`

High-level wrappers that compute multiple metrics at once.

- `evaluate_classification`, `evaluate_multilabel_classification`
- `evaluate_clustering`, `evaluate_clustering_stability` (compares against a random baseline)

### `cvtools.visualization`

Visualization functions.

- `classification`: `display_confusion_matrix`, `plot_roc_curve`, `plot_roc_curves`, `plot_roc_curves_with_confidence_intervals`.
- `clustering`: `visualize_cluster_variability`, `visualize_clustering_stability`.
- `features`: `visualize_features` (PCA or t-SNE), `all_feature_visualizations`, `visualize_feature_distribution`.
- `pytorch.gradcam`: `GradCAMPyTorch` with `get_gradcam_heatmap` and `visualize_heatmap`.
- `pytorch.tensors`: `visualize_tensor`.
- `utils`: `plt_save_and_show`.

### `cvtools.utils`

Other useful functions.

- `array`: `pad_array_to_shape`, `group_arrays_by_shape`, `pad_arrays_to_uniform_size`.
- `random`: `stratified_sampling_by_class`.
- `pytorch.datasets`: `dataset_train_test_split` (stratified, with a per-class minimum), `InfiniteDataLoader`.
- `pytorch.early_stopping`: `EarlyStopping` (`step`, `restore`).
- `pytorch.image_tensors`: `concat_channels_interleaved`, `concat_channels_interleaved_nway`.
- `pytorch.classification`: `topk_predictions`.
- `pytorch.random`: `set_seed`.

---

## Contributing

We welcome contributions! If you’d like to add functionality, report bugs, or improve documentation, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

