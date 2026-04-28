"""
Data modules for AnyMC3D.

A single generic dataset (ClassificationDataset) and matching DataModule
cover PDCAD, BMLMPS, meningioma, and any future 3D volume classification
dataset whose .npy files follow the ``{data_root}/{case_id}_0000.npy``
convention. All dataset-specific behaviour is driven by the Hydra data
config — no per-dataset Python subclass needed.

Augmentation lives in data_augmentation.py and is attached externally
via apply_augmentation(dm, ...).
"""

from .classification_dataset import (
    ClassificationDataset,
    ClassificationDataModule,
)
from .data_augmentation import (
    IMAGE_KEY,
    TransformedDataset,
    apply_augmentation,
    build_train_transforms,
    build_val_transforms,
)

__all__ = [
    "ClassificationDataset",
    "ClassificationDataModule",
    "IMAGE_KEY",
    "TransformedDataset",
    "apply_augmentation",
    "build_train_transforms",
    "build_val_transforms",
]