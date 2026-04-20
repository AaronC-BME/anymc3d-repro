"""
PDCAD NM Dataset for AnyMC3D (v4 — transforms decoupled)
========================================================
This module handles pure data loading for the PDCAD NM binary classification
task. Augmentation lives in ``data_augmentation.py`` and is attached to the
data module externally (e.g. via ``apply_augmentation`` in ``train.py``).

Responsibilities
----------------
    PDCADDataset    : load and resize a pre-normalized ``.npy`` volume.
                      Returns (volume, label, case_id) with NO augmentation.
    PDCADDataModule : build train/val/test DataLoaders, optionally wrapping
                      the base dataset with a ``TransformedDataset`` when
                      ``train_transform`` / ``eval_transform`` attributes
                      have been set.

Volume convention
-----------------
    Input  .npy: (1, 300, 300, 70) float32, already in [0, 1]
    Output:      (1, 308, 308, 70) float32, in [0, 1]

Sanity check
------------
    Run as a module from the project root:
        python -m data_modules.pdcad_dataset
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data_augmentation import TransformedDataset


# ---------------------------------------------------------------
# Raw dataset — no augmentation, no transforms
# ---------------------------------------------------------------

class PDCADDataset(Dataset):
    """
    PyTorch Dataset for PDCAD NM binary classification.

    Loads a pre-normalized ``.npy`` volume, resizes it to ``patch_size`` via
    trilinear interpolation, and returns it alongside the integer label and
    case id. No MONAI transforms are applied here — that is the job of
    ``TransformedDataset`` (see ``data_augmentation.py``).

    Args:
        data_root:   Path to dataset root containing pre-normalized .npy volumes
        labels_path: Path to labels.json (case_id -> 0/1)
        splits_path: Path to splits.json
        split:       One of 'train', 'val', 'test'
        fold:        Fold index (default 0)
        patch_size:  Target volume size [H, W, S] (default [308, 308, 70])
    """

    def __init__(
        self,
        data_root:   str,
        labels_path: str,
        splits_path: str,
        split:       str  = "train",
        fold:        int  = 0,
        patch_size:  list = None,
    ):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.data_root  = Path(data_root)
        self.split      = split
        self.patch_size = patch_size or [308, 308, 70]

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
        return volume, label, case_id


# ---------------------------------------------------------------
# Data module — transforms attached externally via attributes
# ---------------------------------------------------------------

class PDCADDataModule:
    """
    Convenience wrapper that creates train/val/test DataLoaders over the raw
    ``PDCADDataset``. Augmentation is decoupled: set ``train_transform`` and
    ``eval_transform`` after construction (typically via
    ``data_augmentation.apply_augmentation``), and each dataloader method
    wraps its base dataset with ``TransformedDataset`` using those attributes.

    If the transforms are left as ``None`` (the default), the module returns
    raw volumes — useful for profiling or debugging the loading path
    independently from augmentation.
    """

    def __init__(
        self,
        data_root:   str,
        labels_path: str,
        splits_path: str,
        fold:        int  = 0,
        batch_size:  int  = 2,
        num_workers: int  = 8,
        patch_size:  list = None,
    ):
        self.data_root   = data_root
        self.labels_path = labels_path
        self.splits_path = splits_path
        self.fold        = fold
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.patch_size  = patch_size or [308, 308, 70]

        # Transform hooks — set by apply_augmentation() in train.py.
        # Remain None until then, in which case loaders return raw volumes.
        self.train_transform = None
        self.eval_transform  = None

    # ── Base-dataset factory (no transform) ──────────────────────────────────
    def _make_dataset(self, split: str) -> PDCADDataset:
        return PDCADDataset(
            data_root   = self.data_root,
            labels_path = self.labels_path,
            splits_path = self.splits_path,
            split       = split,
            fold        = self.fold,
            patch_size  = self.patch_size,
        )

    # ── DataLoader builders — wrap with TransformedDataset on the fly ────────
    def train_dataloader(self) -> DataLoader:
        ds = self._make_dataset("train")
        ds = TransformedDataset(ds, self.train_transform)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True,
                          drop_last=True)

    def val_dataloader(self) -> DataLoader:
        ds = self._make_dataset("val")
        ds = TransformedDataset(ds, self.eval_transform)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self) -> DataLoader:
        ds = self._make_dataset("test")
        ds = TransformedDataset(ds, self.eval_transform)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    # Run from project root with:
    #     python -m data_modules.pdcad_dataset
    from .data_augmentation import apply_augmentation

    DATA_ROOT   = "/home/jma/Documents/projects/aaron/AnyMC3D/preprocessed_data/PDCAD"
    LABELS_PATH = "/home/jma/Documents/projects/aaron/nnssl_dataset/nnssl_preprocessed/Dataset009_PDCAD_NM/labels.json"
    SPLITS_PATH = "/home/jma/Documents/projects/aaron/nnssl_dataset/nnssl_preprocessed/Dataset009_PDCAD_NM/splits.json"

    print("=" * 60)
    print("PDCADDataset v4 Sanity Check (transforms decoupled)")
    print("=" * 60)

    dm = PDCADDataModule(
        data_root   = DATA_ROOT,
        labels_path = LABELS_PATH,
        splits_path = SPLITS_PATH,
        batch_size  = 2,
        num_workers = 2,
        patch_size  = [308, 308, 70],
    )
    apply_augmentation(dm, augment_train=True)

    loader = dm.train_dataloader()
    vols, labels, ids = next(iter(loader))
    print(f"  Batch volumes: {vols.shape}")
    print(f"  Batch labels:  {labels}")
    print(f"  Case IDs:      {list(ids)}")
    print("\nAll checks passed!")