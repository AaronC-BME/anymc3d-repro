"""
Data modules for AnyMC3D.

Each submodule pairs a raw Dataset (loading + resize only) with a
DataModule that builds train/val/test DataLoaders. Augmentation lives
in data_augmentation.py and is attached externally.
"""

from .pdcad_dataset import PDCADDataset, PDCADDataModule
from .data_augmentation import (
    IMAGE_KEY,
    TransformedDataset,
    apply_augmentation,
    build_train_transforms,
    build_val_transforms,
)

__all__ = [
    "PDCADDataset",
    "PDCADDataModule",
    "IMAGE_KEY",
    "TransformedDataset",
    "apply_augmentation",
    "build_train_transforms",
    "build_val_transforms",
]