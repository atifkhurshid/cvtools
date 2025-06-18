# Computer Vision Tools

A collection of useful Python functions for common computer vision tasks, designed to support research, prototyping, and production pipelines. The toolkit provides high-level utilities for image preprocessing, feature decomposition, dataset management, and visualization.

---

## Modules

### `datasets`
> **Purpose:** Streamlined dataset loading and integration with PyTorch and TensorFlow for classification tasks.

The `datasets` module provides standardized dataset loaders for image classification problems, including both generic loaders and specialized datasets. It includes native support for **PyTorch** and **TensorFlow** frameworks, enabling seamless integration into deep learning pipelines.

#### Structure

- **Base Module:** `cvtools.datasets.classification`  
  Contains core dataset classes for handling image folders and specific datasets.

- **Framework-Specific Wrappers:**
  - **PyTorch:** `cvtools.datasets.classification.pytorch`  
    All PyTorch-compatible datasets have a `PT` suffix.
  - **TensorFlow:** `cvtools.datasets.classification.tensorflow`  
    All TensorFlow-compatible datasets have a `TF` suffix.

---

#### Available Datasets & Loaders

##### 1. `ClassificationDataset`
A flexible and generic dataset loader for image classification tasks where:
- Each class is represented as a folder.
- Labels are inferred from folder names.
- Can be used for custom or small-scale datasets.

**Wrappers:**
- `ClassificationDatasetPT`: PyTorch `torch.utils.data.Dataset` class.
- `ClassificationDatasetTF`: TensorFlow `tf.keras.utils.Sequence` generator.

---

##### 2. `CXRDataset`
A dataset loader for the NIH Chest X-Ray Dataset, designed for multi-label classification of radiographic images.

**Wrappers:**
- `CXRDatasetPT`: PyTorch-compatible version.

---

##### 3. `SUNDataset`
A dataset loader for the **SUN database** (Scene Understanding) used for scene categorization.

**Wrappers:**
- `SUNDatasetPT`: PyTorch-compatible version.

> TensorFlow wrapper is not yet implemented.

---

#### Example Usage

```python
# Load a custom classification dataset with TensorFlow
from cvtools.datasets.classification.tensorflow import ClassificationDatasetTF
dataset = ClassificationDatasetTF(root="/path/to/images", batch_size=32)

# Load NIH Chest X-Ray dataset
from cvtools.datasets.classification import CXRDataset
cxr_dataset = CXRDatasetPT(root_dir='/path/to/dataset', image_size=(224, 224), train=True)

# Load SUN database with PyTorch
from cvtools.datasets.classification.pytorch import SUNDatasetPT
sun_dataset = SUNDatasetPT(root_dir='/path/to/dataset', transforms=...)
```

#### Extensibility

This module is designed for easy extension:
- Create a new folder for the dataset in `cvtools.datasets.classification`.
- Add a new base dataset class.
- Add framework-specific wrappers in the same folder using the naming convention:
    - `DatasetNamePT` for PyTorch
    - `DatasetNameTF` for TensorFlow
- Add import statements for framework-specific wrappers in `cvtools.datasets.classification.pytorch.py` and `cvtools.datasets.classification.tensorflow.py` files.

### `decomposition`
> **Purpose:** Dimensionality reduction and feature analysis.

Contains customized implementations of **Principal Component Analysis (PCA)** and related decomposition techniques for feature reduction, noise filtering, and exploratory data analysis.

### `image`
> **Purpose:** Efficient image input/output and preprocessing.

Includes image reading, saving, format conversion, resizing, normalization, and color manipulation utilities. These functions streamline data preparation and image pipeline operations.

### `visualization`
> **Purpose:** Visual insight into model behavior and features.

Supports techniques for feature visualization, activation map generation, and overlay tools to understand model attention and feature extraction.

### `utils`
> **Purpose:** Miscellaneous helpers for various CV tasks.

A collection of utility functions such as file handling, logging, metric computation, and other general-purpose routines that are widely applicable across modules.

---

## Installation

Clone the repository and install it in your Python environment in *editable* mode using:

```bash
git clone https://github.com/yourusername/cvtools.git
pip install -e /path/to/cvtools/
```

## Usage



## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Pillow
- tqdm
- scikit-learn
- PyTorch or TensorFlow (depending on your workflow)

## Contributing

We welcome contributions! If you’d like to add functionality, report bugs, or improve documentation, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

