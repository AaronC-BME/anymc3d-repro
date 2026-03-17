"""
PDCAD NM Dataset for AnyMC3D (v3 — MONAI Augmentation)
=======================================================
Uses MONAI transforms for nnU-Net-style data augmentation,
matching AnyMC3D paper Appendix A3.

Expects .npy volumes preprocessed by preprocess_anymc3d_v3.py
(MONAI ScaleIntensityRangePercentiled, already in [0, 1]).

Augmentation pipeline (from paper Appendix A3):
  1.  Random flips along all 3 spatial axes (p=0.5 each)
  2.  Random rotation ±30° per axis (p=0.2)
  3.  Random zoom 0.7–1.4× (p=0.2)
  4.  Random affine translation ±10 voxels (p=0.2)
  5.  Gaussian noise σ=0.02 (p=0.25)
  6.  Gaussian blur σ=0.5–1.0 (p=0.2)
  7.  Brightness multiplication 0.75–1.25× (p=0.15)
  8.  Contrast augmentation 0.75–1.25× (p=0.15)
  9.  Low-resolution simulation 0.5–1.0× (p=0.2)
  10. Gamma correction γ=0.7–1.5 (p=0.25)
"""

import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from monai.transforms import (
    Compose,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandGibbsNoised,
    ScaleIntensityd,
    EnsureTyped,
    RandLambdad,
)

IMAGE_KEY = "image"


def build_train_transforms():
    """
    Build MONAI augmentation pipeline matching nnU-Net-style augmentation
    from AnyMC3D paper Appendix A3.

    Input:  dict with IMAGE_KEY -> Tensor (1, H, W, S) in [0, 1]
    Output: dict with IMAGE_KEY -> Tensor (1, H, W, S) in [0, 1]
    """
    return Compose([
        # 1. Random flips along all 3 spatial axes, p=0.5 each
        RandFlipd(keys=IMAGE_KEY, prob=0.5, spatial_axis=0),
        RandFlipd(keys=IMAGE_KEY, prob=0.5, spatial_axis=1),
        RandFlipd(keys=IMAGE_KEY, prob=0.5, spatial_axis=2),

        # 2–4. Random rotation (±30°), zoom (0.7–1.4×), translation (±10 voxels)
        # Combined in a single RandAffined call for efficiency
        RandAffined(
            keys=IMAGE_KEY,
            prob=0.2,
            rotate_range=(0.5236, 0.5236, 0.5236),  # ±30° in radians
            scale_range=(-0.3, 0.4),                  # scale 0.7–1.4×
            translate_range=(10, 10, 10),              # ±10 voxels
            mode="bilinear",
            padding_mode="border",
        ),

        # 5. Gaussian noise, p=0.25
        RandGaussianNoised(
            keys=IMAGE_KEY,
            prob=0.25,
            mean=0.0,
            std=0.02,
        ),

        # 6. Gaussian blur σ=0.5–1.0, p=0.2
        RandGaussianSmoothd(
            keys=IMAGE_KEY,
            prob=0.2,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
        ),

        # 7. Brightness multiplication 0.75–1.25×, p=0.15
        RandScaleIntensityd(
            keys=IMAGE_KEY,
            prob=0.15,
            factors=(-0.25, 0.25),  # multiplies by (1 + factor)
        ),

        # 8. Contrast augmentation, p=0.15
        RandAdjustContrastd(
            keys=IMAGE_KEY,
            prob=0.15,
            gamma=(0.75, 1.25),
        ),

        # 9. Low-resolution simulation (downsample then upsample), p=0.2
        RandLambdad(
            keys=IMAGE_KEY,
            prob=0.2,
            func=_low_resolution_simulation,
        ),

        # 10. Gamma correction γ=0.7–1.5, p=0.25
        RandLambdad(
            keys=IMAGE_KEY,
            prob=0.25,
            func=_gamma_correction,
        ),

        # Clamp to [0, 1] and ensure tensor type
        RandLambdad(keys=IMAGE_KEY, prob=1.0, func=lambda x: x.clamp(0, 1) if isinstance(x, torch.Tensor) else np.clip(x, 0, 1)),
        EnsureTyped(keys=IMAGE_KEY, dtype=torch.float32),
    ])


def _low_resolution_simulation(x):
    """
    Low-resolution simulation: downsample by random factor 0.5–1.0,
    then upsample back to original size.
    Matches paper Appendix A3 item 9.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    scale = float(torch.empty(1).uniform_(0.5, 1.0).item())
    orig_shape = x.shape[1:]  # (H, W, S) — skip channel dim
    x_5d = x.unsqueeze(0)     # (1, C, H, W, S)
    x_5d = F.interpolate(x_5d, scale_factor=scale, mode='trilinear', align_corners=False)
    x_5d = F.interpolate(x_5d, size=orig_shape, mode='trilinear', align_corners=False)
    return x_5d.squeeze(0).clamp(0, 1)


def _gamma_correction(x):
    """
    Gamma correction γ=0.7–1.5, with optional inversion (p=0.15).
    Matches paper Appendix A3 item 10.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    gamma = float(torch.empty(1).uniform_(0.7, 1.5).item())
    invert = torch.rand(1).item() < 0.15
    if invert:
        x = 1.0 - x
    x = x.clamp(min=0).pow(gamma)
    if invert:
        x = 1.0 - x
    return x.clamp(0, 1)


def build_val_transforms():
    """No augmentation for val/test — just ensure tensor type."""
    return Compose([
        EnsureTyped(keys=IMAGE_KEY, dtype=torch.float32),
    ])


class PDCADDataset(Dataset):
    """
    PyTorch Dataset for PDCAD NM binary classification.

    Args:
        data_root:   Path to dataset root containing pre-normalized .npy volumes
        labels_path: Path to labels.json (case_id -> 0/1)
        splits_path: Path to splits.json
        split:       One of 'train', 'val', 'test'
        fold:        Fold index (default 0)
        augment:     Whether to apply data augmentation (train only)
        patch_size:  Target volume size [H, W, S] (default [308, 308, 70])
    """

    def __init__(
        self,
        data_root:   str,
        labels_path: str,
        splits_path: str,
        split:       str  = "train",
        fold:        int  = 0,
        augment:     bool = False,
        patch_size:  list = None,
    ):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.data_root  = Path(data_root)
        self.split      = split
        self.augment    = augment
        self.patch_size = patch_size or [308, 308, 70]

        # Build MONAI transforms
        if augment and split == "train":
            self.transform = build_train_transforms()
        else:
            self.transform = build_val_transforms()

        with open(labels_path) as f:
            self.labels = json.load(f)

        with open(splits_path) as f:
            splits = json.load(f)

        self.case_ids = splits[str(fold)][split]

        missing = [c for c in self.case_ids if c not in self.labels]
        if missing:
            print(f"WARNING: {len(missing)} cases in {split} split have no label, skipping:")
            for m in missing:
                print(f"  {m}")
        self.case_ids = [c for c in self.case_ids if c in self.labels]

        print(f"PDCADDataset [{split}]: {len(self.case_ids)} cases")
        print(f"  patch_size: {self.patch_size}")
        print(f"  augmentation: {'MONAI nnU-Net-style' if (augment and split == 'train') else 'none'}")
        self._print_class_distribution()

    def _print_class_distribution(self):
        from collections import Counter
        counts = Counter(self.labels[c] for c in self.case_ids)
        class_names = {0: "Class0", 1: "Class1"}
        dist = {class_names[k]: counts.get(k, 0) for k in sorted(class_names)}
        print(f"  Class distribution: {dist}")

    def _get_npy_path(self, case_id: str) -> Path:
        return self.data_root / f"{case_id}_0000.npy"

    def _load_volume(self, case_id: str) -> torch.Tensor:
        """
        Load pre-normalized volume and resize to patch_size.

        Input .npy:  (1, 300, 300, 70) float32, already in [0, 1]
        Output:      (1, 308, 308, 70) float32, in [0, 1]
        """
        path = self._get_npy_path(case_id)
        if not path.exists():
            raise FileNotFoundError(f"Volume not found: {path}")

        arr    = np.load(str(path))                  # (1, H, W, S) float32, [0, 1]
        volume = torch.from_numpy(arr).float()

        # Resize to patch_size
        target_H, target_W, target_S = self.patch_size
        _, cur_H, cur_W, cur_S = volume.shape

        if (cur_H != target_H) or (cur_W != target_W) or (cur_S != target_S):
            volume = volume.unsqueeze(0)  # (1, 1, H, W, S)
            volume = F.interpolate(
                volume,
                size=(target_H, target_W, target_S),
                mode='trilinear',
                align_corners=False,
            )
            volume = volume.squeeze(0)    # (1, H, W, S)
            volume = volume.clamp(0.0, 1.0)

        return volume

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        """
        Returns:
            volume:  Tensor (1, 308, 308, 70) in [0, 1]
            label:   Tensor scalar int64
            case_id: str
        """
        case_id = self.case_ids[idx]
        volume  = self._load_volume(case_id)
        label   = torch.tensor(self.labels[case_id], dtype=torch.long)

        # Apply MONAI transforms (augmentation for train, identity for val/test)
        data_dict = self.transform({IMAGE_KEY: volume})
        volume = data_dict[IMAGE_KEY]

        return volume, label, case_id


class PDCADDataModule:
    """
    Convenience wrapper that creates train/val/test DataLoaders.

    Args match pdcad.yaml config:
        batch_size:  2  (from train.batch_size)
        num_workers: 8  (from train.num_workers)
        patch_size:  [308, 308, 70] (from train.patch_size)
    """

    def __init__(
        self,
        data_root:   str,
        labels_path: str,
        splits_path: str,
        fold:        int  = 0,
        batch_size:  int  = 2,
        num_workers: int  = 8,
        augment:     bool = True,
        patch_size:  list = None,
    ):
        self.data_root   = data_root
        self.labels_path = labels_path
        self.splits_path = splits_path
        self.fold        = fold
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.augment     = augment
        self.patch_size  = patch_size or [308, 308, 70]

    def _make_dataset(self, split: str, augment: bool = False) -> PDCADDataset:
        return PDCADDataset(
            data_root   = self.data_root,
            labels_path = self.labels_path,
            splits_path = self.splits_path,
            split       = split,
            fold        = self.fold,
            augment     = augment,
            patch_size  = self.patch_size,
        )

    def train_dataloader(self) -> DataLoader:
        ds = self._make_dataset("train", augment=self.augment)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True,
                          drop_last=True)

    def val_dataloader(self) -> DataLoader:
        ds = self._make_dataset("val", augment=False)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        ds = self._make_dataset("test", augment=False)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    DATA_ROOT   = "/home/jma/Documents/projects/aaron/AnyMC3D/preprocessed_data/PDCAD"
    LABELS_PATH = "/home/jma/Documents/projects/aaron/nnssl_dataset/nnssl_preprocessed/Dataset009_PDCAD_NM/labels.json"
    SPLITS_PATH = "/home/jma/Documents/projects/aaron/nnssl_dataset/nnssl_preprocessed/Dataset009_PDCAD_NM/splits.json"

    print("=" * 60)
    print("PDCADDataset v3 Sanity Check (MONAI Augmentation)")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        print(f"\n--- {split} ---")
        ds = PDCADDataset(
            data_root   = DATA_ROOT,
            labels_path = LABELS_PATH,
            splits_path = SPLITS_PATH,
            split       = split,
            augment     = (split == "train"),
            patch_size  = [308, 308, 70],
        )
        vol, label, case_id = ds[0]
        print(f"  First case:    {case_id}")
        print(f"  Volume shape:  {vol.shape}")    # expect (1, 308, 308, 70)
        print(f"  Volume dtype:  {vol.dtype}")
        print(f"  Min/Max:       {vol.min():.3f} / {vol.max():.3f}")
        print(f"  Label:         {label.item()}")

    print("\n--- DataLoader test ---")
    dm = PDCADDataModule(
        data_root   = DATA_ROOT,
        labels_path = LABELS_PATH,
        splits_path = SPLITS_PATH,
        batch_size  = 2,
        num_workers = 2,
        patch_size  = [308, 308, 70],
    )
    loader = dm.train_dataloader()
    vols, labels, ids = next(iter(loader))
    print(f"  Batch volumes: {vols.shape}")   # expect [2, 1, 308, 308, 70]
    print(f"  Batch labels:  {labels}")
    print(f"  Case IDs:      {list(ids)}")
    print("\nAll checks passed!")