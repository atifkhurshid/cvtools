"""
Preprocessing functions for the SIIM-FISABIO-RSNA COVID-19 Detection Challenge dataset.
"""

# Author: Atif Khurshid
# Created: 2026-09-23
# Modified: None
# Version: 1.0
# Changelog:
#     - 2026-09-23: Initial version.

import os
import shutil

import numpy as np

from tqdm import tqdm
from scipy import ndimage

from ....image import imresize_minimum, imwrite


def _read_xray(
        path: str,
        voi_lut: bool = True,
        fix_monochrome: bool = True
    ) -> np.ndarray:
    """
    Read a DICOM file and convert it to a numpy array.

    Adapted from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way

    Parameters
    ----------
    path : str
        Path to the DICOM file.
    voi_lut : bool, optional
        Whether to apply the VOI LUT (Value of Interest Look-Up Table) transformation.
    fix_monochrome : bool, optional
        Whether to fix the monochrome images that are inverted (MONOCHROME1).
    
    Returns
    -------
    np.ndarray
        The image as a 2D numpy array, normalized to the range [0, 255] and of type uint8.
    """
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    dicom = pydicom.dcmread(path)
    pixel_array = dicom.pixel_array.astype(np.float64)

    # Apply rescale slope/intercept if present (converts stored values
    # to actual measured values, e.g. Hounsfield units for CT).
    slope = float(getattr(dicom, "RescaleSlope", 1))
    intercept = float(getattr(dicom, "RescaleIntercept", 0))
    pixel_array = pixel_array * slope + intercept

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to 
    # "human-friendly" view
    if voi_lut:
        pixel_array = apply_voi_lut(pixel_array, dicom)
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array
        
    # Normalize to 0-255 (8-bit) range.
    lo = np.percentile(pixel_array, 0.5)
    hi = np.percentile(pixel_array, 99.5)

    pixel_array = np.clip(pixel_array, lo, hi)
    pixel_array = pixel_array - lo
    if hi != lo:
        pixel_array = pixel_array / (hi - lo)
    pixel_array = (pixel_array * 255.0).astype(np.uint8)
        
    return pixel_array


def _crop_padding(
        image: np.ndarray,
        threshold: int = 10,
        min_area_fraction: float = 0.01
    ) -> np.ndarray:
    """
    Crop black padding borders by finding the largest connected
    region of non-padding content, ignoring isolated noise pixels.

    Parameters
    ----------
    image : np.ndarray
        Input 2D image array.
    threshold : int, optional
        Pixel values at or below this are treated as padding/background.
    min_area_fraction : float, optional
        Minimum size (as a fraction of total image area) for a connected component
        to be considered real content rather than noise.
    
    Returns
    -------
    np.ndarray
        The cropped image.
    """
    mask = image > threshold

    # Remove isolated noise pixels/small clusters before labeling.
    # A binary opening erodes then dilates, which strips out specks
    # that aren't part of a larger connected region.
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))

    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return image  # Nothing above threshold; nothing to crop.

    # Find the largest connected component by pixel count.
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    largest_label = np.argmax(sizes) + 1

    min_area = min_area_fraction * image.size
    if sizes[largest_label - 1] < min_area:
        return image  # Largest region too small; likely all noise.

    largest_mask = labeled == largest_label
    rows = np.any(largest_mask, axis=1)
    cols = np.any(largest_mask, axis=0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]

    return image[row_min:row_max + 1, col_min:col_max + 1]


def preprocess_dataset(
        src_root: str,
        dst_root: str,
        target_size: int = 512    
    ):
    """
    Preprocess the SIIM-FISABIO-RSNA COVID-19 Detection Challenge dataset by
    converting DICOM files to PNG format, followed by cropping and resizing the images.
    The labels CSV files (_study_level.csv and _image_level.csv) are also copied to the destination directory.

    Parameters
    ----------
    src_root : str
        Path to the root directory containing the DICOM files.
    dst_root : str
        Path to the root directory where the converted PNG files will be saved.
    target_size : int, optional
        The size of the output images (default is 512).
    """
    dicom_paths = []
    for dirpath, _, filenames in os.walk(src_root):
        for filename in filenames:
            if filename.lower().endswith('.dcm'):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, src_root)
                dicom_paths.append(rel_path)

    for rel_path in tqdm(dicom_paths, desc="Converting DICOM to PNG", total=len(dicom_paths)):

        src_path = os.path.join(src_root, rel_path)
        dst_path = os.path.join(dst_root, os.path.splitext(rel_path)[0] + '.png')
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        try:
            image = _read_xray(src_path)
            image = imresize_minimum(image, target_size)
            image = _crop_padding(image)
            imwrite(dst_path, image)

        except Exception as e:
            print(f"Failed to convert {src_path}: {e}")

    # Copy labels csv to the destination directory
    for filename in ['_study_level.csv', '_image_level.csv']:
        src_csv_path = os.path.join(src_root, filename)
        dst_csv_path = os.path.join(dst_root, filename)
        if os.path.exists(src_csv_path):
            shutil.copy(src_csv_path, dst_csv_path)
