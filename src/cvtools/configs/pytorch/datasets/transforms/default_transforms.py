"""
Default torchvision transforms.
"""

# Author: Atif Khurshid
# Created: 2026-08-04
# Modified: 2026-10-22
# Version: 1.1
# Changelog:
#     - 2026-08-04: Initial version.
#     - 2026-10-22: Refactored code into separate files for better organization.

import torchvision.transforms.v2 as T

from .register import register_transform


register_transform("resize")(T.Resize)
register_transform("scale_jitter")(T.ScaleJitter)
register_transform("random_shortest_size")(T.RandomShortestSize)
register_transform("random_resize")(T.RandomResize)
register_transform("random_crop")(T.RandomCrop)
register_transform("random_resized_crop")(T.RandomResizedCrop)
register_transform("random_iou_crop")(T.RandomIoUCrop)
register_transform("center_crop")(T.CenterCrop)
register_transform("five_crop")(T.FiveCrop)
register_transform("ten_crop")(T.TenCrop)
register_transform("random_horizontal_flip")(T.RandomHorizontalFlip)
register_transform("random_vertical_flip")(T.RandomVerticalFlip)
register_transform("pad")(T.Pad)
register_transform("random_zoom_out")(T.RandomZoomOut)
register_transform("random_rotation")(T.RandomRotation)
register_transform("random_affine")(T.RandomAffine)
register_transform("random_perspective")(T.RandomPerspective)
register_transform("elastic_transform")(T.ElasticTransform)
register_transform("color_jitter")(T.ColorJitter)
register_transform("random_channel_permutation")(T.RandomChannelPermutation)
register_transform("random_photometric_distort")(T.RandomPhotometricDistort)
register_transform("grayscale")(T.Grayscale)
register_transform("rgb")(T.RGB)
register_transform("random_grayscale")(T.RandomGrayscale)
register_transform("gaussian_blur")(T.GaussianBlur)
register_transform("gaussian_noise")(T.GaussianNoise)
register_transform("random_invert")(T.RandomInvert)
register_transform("random_posterize")(T.RandomPosterize)
register_transform("random_solarize")(T.RandomSolarize)
register_transform("random_adjust_sharpness")(T.RandomAdjustSharpness)
register_transform("random_autocontrast")(T.RandomAutocontrast)
register_transform("random_equalize")(T.RandomEqualize)
register_transform("normalize")(T.Normalize)
register_transform("random_erasing")(T.RandomErasing)
register_transform("Lambda")(T.Lambda)
register_transform("jpeg")(T.JPEG)
register_transform("to_image")(T.ToImage)
register_transform("to_pure_tensor")(T.ToPureTensor)
register_transform("pil_to_tensor")(T.PILToTensor)
register_transform("to_pil_image")(T.ToPILImage)
register_transform("to_dtype")(T.ToDtype)
