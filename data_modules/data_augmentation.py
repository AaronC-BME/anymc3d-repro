"""
Data Augmentation for PDCAD NM (AnyMC3D)
=========================================
MONAI-based augmentation pipeline matching AnyMC3D paper Appendix A3
(nnU-Net-style augmentation for 3D medical volumes).

Public API
----------
    IMAGE_KEY              : str, the dict key used by the MONAI pipeline
    build_train_transforms : Compose of all training-time augmentations
    build_val_transforms   : Compose for val/test (identity, tensor cast only)
    TransformedDataset     : thin wrapper that applies a transform to a
                             base dataset at __getitem__ time
    apply_augmentation     : attach train/eval transforms to a data module
                             that exposes `train_transform`/`eval_transform`
                             attributes

Pipeline (paper Appendix A3)
----------------------------
    1.  Random flips along all 3 spatial axes        (p=0.5 each)
    2.  Random rotation ±30° per axis                (p=0.2)
    3.  Random zoom 0.7–1.4×                         (p=0.2)
    4.  Random affine translation ±10 voxels         (p=0.2)
    5.  Gaussian noise σ=0.02                        (p=0.25)
    6.  Gaussian blur σ=0.5–1.0                      (p=0.2)
    7.  Brightness multiplication 0.75–1.25×         (p=0.15)
    8.  Contrast augmentation 0.75–1.25×             (p=0.15)
    9.  Low-resolution simulation 0.5–1.0×           (p=0.2)
    10. Gamma correction γ=0.7–1.5                   (p=0.25)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from monai.transforms import (
    Compose,
    EnsureTyped,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandLambdad,
    RandScaleIntensityd,
)

# ---------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------

IMAGE_KEY = "image"


# ---------------------------------------------------------------
# Private helpers for items 9 & 10 (no direct MONAI equivalent)
# ---------------------------------------------------------------

def _low_resolution_simulation(x):
    """Downsample 0.5–1.0× then upsample back (Appendix A3, item 9)."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    scale       = float(torch.empty(1).uniform_(0.5, 1.0).item())
    orig_shape  = x.shape[1:]                      # (H, W, S)
    x_5d        = x.unsqueeze(0)                   # (1, C, H, W, S)
    x_5d        = F.interpolate(x_5d, scale_factor=scale,
                                mode="trilinear", align_corners=False)
    x_5d        = F.interpolate(x_5d, size=orig_shape,
                                mode="trilinear", align_corners=False)
    return x_5d.squeeze(0).clamp(0, 1)


def _gamma_correction(x):
    """γ ∈ [0.7, 1.5] with optional inversion p=0.15 (Appendix A3, item 10)."""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    gamma  = float(torch.empty(1).uniform_(0.7, 1.5).item())
    invert = torch.rand(1).item() < 0.15

    if invert:
        x = 1.0 - x
    x = x.clamp(min=0).pow(gamma)
    if invert:
        x = 1.0 - x

    return x.clamp(0, 1)


def _clamp_01(x):
    """Post-augmentation safety clamp back into [0, 1]."""
    if isinstance(x, torch.Tensor):
        return x.clamp(0, 1)
    return np.clip(x, 0, 1)


# ---------------------------------------------------------------
# Public transform builders
# ---------------------------------------------------------------

def build_train_transforms() -> Compose:
    """Full nnU-Net-style augmentation pipeline for training."""
    return Compose([
        RandFlipd(keys=IMAGE_KEY, prob=0.5, spatial_axis=0),
        RandFlipd(keys=IMAGE_KEY, prob=0.5, spatial_axis=1),
        RandFlipd(keys=IMAGE_KEY, prob=0.5, spatial_axis=2),

        RandAffined(
            keys            = IMAGE_KEY,
            prob            = 0.2,
            rotate_range    = (0.5236, 0.5236, 0.5236),   # ±30°
            scale_range     = (-0.3, 0.4),                # 0.7–1.4×
            translate_range = (10, 10, 10),               # ±10 voxels
            mode            = "bilinear",
            padding_mode    = "border",
        ),

        RandGaussianNoised(keys=IMAGE_KEY, prob=0.25, mean=0.0, std=0.02),

        RandGaussianSmoothd(
            keys    = IMAGE_KEY, prob = 0.2,
            sigma_x = (0.5, 1.0), sigma_y = (0.5, 1.0), sigma_z = (0.5, 1.0),
        ),

        RandScaleIntensityd(keys=IMAGE_KEY, prob=0.15, factors=(-0.25, 0.25)),
        RandAdjustContrastd(keys=IMAGE_KEY, prob=0.15, gamma=(0.75, 1.25)),

        RandLambdad(keys=IMAGE_KEY, prob=0.2,  func=_low_resolution_simulation),
        RandLambdad(keys=IMAGE_KEY, prob=0.25, func=_gamma_correction),

        RandLambdad(keys=IMAGE_KEY, prob=1.0,  func=_clamp_01),
        EnsureTyped(keys=IMAGE_KEY, dtype=torch.float32),
    ])


def build_val_transforms() -> Compose:
    """No-op transform for val/test — ensures float32 tensor type only."""
    return Compose([
        EnsureTyped(keys=IMAGE_KEY, dtype=torch.float32),
    ])


# ---------------------------------------------------------------
# Wrapper dataset — the bridge between raw loading and augmentation
# ---------------------------------------------------------------

class TransformedDataset(Dataset):
    """
    Wraps a base dataset and applies a MONAI dict transform to its volumes.

    The base dataset is expected to return a 3-tuple ``(volume, label, case_id)``
    where ``volume`` is a Tensor of shape ``(1, H, W, S)``. If ``transform`` is
    ``None`` this acts as a pass-through.

    This separation means the base dataset (``PDCADDataset``) only has to know
    how to load and resize volumes — augmentation is attached externally.
    """

    def __init__(self, base_dataset: Dataset, transform: Compose | None = None):
        self.base      = base_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        volume, label, case_id = self.base[idx]
        if self.transform is not None:
            volume = self.transform({IMAGE_KEY: volume})[IMAGE_KEY]
        return volume, label, case_id


# ---------------------------------------------------------------
# Wiring helper — keeps train.py tiny
# ---------------------------------------------------------------

def apply_augmentation(datamodule, augment_train: bool = True):
    """
    Attach training and eval transforms to a data module that exposes
    ``train_transform`` and ``eval_transform`` attributes.

    The data module is responsible for actually wrapping its datasets with
    ``TransformedDataset`` using those attributes inside its dataloader
    methods — this helper only decides *which* transforms go where.

    Args:
        datamodule:    a data module instance (e.g. PDCADDataModule).
        augment_train: if False, training also uses the identity transform
                       (useful for debugging or ablations).

    Returns:
        The same datamodule (mutated), for chaining convenience.
    """
    datamodule.train_transform = (
        build_train_transforms() if augment_train else build_val_transforms()
    )
    datamodule.eval_transform = build_val_transforms()
    return datamodule


# ---------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------

if __name__ == "__main__":
    print("Sanity-checking data_augmentation transforms...")

    vol = torch.rand(1, 308, 308, 70)

    train_tf = build_train_transforms()
    val_tf   = build_val_transforms()

    out_train = train_tf({IMAGE_KEY: vol})[IMAGE_KEY]
    out_val   = val_tf({IMAGE_KEY: vol})[IMAGE_KEY]

    print(f"  train -> shape={tuple(out_train.shape)}  "
          f"range=[{out_train.min():.3f}, {out_train.max():.3f}]")
    print(f"  val   -> shape={tuple(out_val.shape)}  "
          f"range=[{out_val.min():.3f}, {out_val.max():.3f}]")

    # Wrapper smoke-test with a toy base dataset
    class _Dummy(Dataset):
        def __len__(self): return 3
        def __getitem__(self, i):
            return torch.rand(1, 308, 308, 70), torch.tensor(i % 2), f"case_{i}"

    wrapped = TransformedDataset(_Dummy(), train_tf)
    v, y, cid = wrapped[0]
    print(f"  wrapper -> volume={tuple(v.shape)}  label={y.item()}  id={cid}")
    print("OK.")