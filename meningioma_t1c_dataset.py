"""
Meningioma Whole-Brain Dataset for AnyMC3D
Loads full T1c volumes (no ROI crop) from Blosc2 format, resizes to (1, 256, 256, 256).

Data format:
    - Volumes: variable shape ~(1, 250, 250, 170) → resized to (1, 256, 256, 256)
    - Labels:  0=MG1, 1=MG2, 2=MG3, 3=MG4
    - Path:    data_root/{case_id}/ses-DEFAULT/{case_id}_0000.b2nd
"""

import json
import torch
import numpy as np
import blosc2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


TARGET_SIZE = 256


class MeningiomaT1cDataset(Dataset):
    """
    PyTorch Dataset for meningioma molecular subtype classification
    using full whole-brain T1c volumes (no ROI crop).

    Args:
        data_root:   Path to dataset root containing per-case folders
        labels_path: Path to labels.json  (case_id -> 0/1/2/3)
        splits_path: Path to splits_hold_out.json
        split:       One of 'train', 'val', 'test'
        fold:        Fold index (default 0)
        augment:     Whether to apply data augmentation (train only)
    """

    def __init__(
        self,
        data_root:   str,
        labels_path: str,
        splits_path: str,
        split:       str  = "train",
        fold:        int  = 0,
        augment:     bool = False,
    ):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.data_root = Path(data_root)
        self.split     = split
        self.augment   = augment

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

        print(f"MeningiomaT1cDataset [{split}]: {len(self.case_ids)} cases")
        self._print_class_distribution()

    def _print_class_distribution(self):
        from collections import Counter
        counts     = Counter(self.labels[c] for c in self.case_ids)
        class_names = {0: "MG1", 1: "MG2", 2: "MG3", 3: "MG4"}
        dist = {class_names[k]: counts.get(k, 0) for k in sorted(class_names)}
        print(f"  Class distribution: {dist}")

    def _get_b2nd_path(self, case_id: str) -> Path:
        """
        Structure: data_root/{case_id}/ses-DEFAULT/{case_id}.b2nd
        The case_id already contains _0000 suffix in the splits/labels JSON,
        so the file is just {case_id}.b2nd directly.
        """
        case_id_dir = case_id.replace("_0000", "")  # remove _0000 suffix if present
        return self.data_root / case_id_dir / "ses-DEFAULT" / f"{case_id}.b2nd"

    def _resize_volume(self, x: torch.Tensor) -> torch.Tensor:
        """
        Resize to (1, TARGET_SIZE, TARGET_SIZE, TARGET_SIZE) via trilinear interpolation.
        x: (1, D, H, W)
        """
        x = x.unsqueeze(0)   # → (1, 1, D, H, W)
        x = F.interpolate(
            x,
            size=(TARGET_SIZE, TARGET_SIZE, TARGET_SIZE),
            mode='trilinear',
            align_corners=False,
        )
        return x.squeeze(0)  # → (1, 256, 256, 256)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Per-volume min-max normalization to [0, 1]."""
        x_min = x.min()
        x_max = x.max()
        if (x_max - x_min) > 1e-8:
            x = (x - x_min) / (x_max - x_min)
        return x

    def _load_volume(self, case_id: str) -> torch.Tensor:
        path = self._get_b2nd_path(case_id)
        if not path.exists():
            raise FileNotFoundError(f"Volume not found: {path}")
        arr    = blosc2.open(str(path))[:]
        arr    = np.ascontiguousarray(arr)
        volume = torch.from_numpy(arr).float().clone()   # (1, D, H, W)
        volume = self._normalize(volume)
        volume = self._resize_volume(volume)             # (1, 256, 256, 256)
        return volume

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple augmentation: random flips + Gaussian noise + brightness scaling.
        x: (1, 256, 256, 256)
        """
        for dim in [1, 2, 3]:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[dim])

        if torch.rand(1).item() < 0.25:
            x = x + torch.randn_like(x) * 0.05

        if torch.rand(1).item() < 0.15:
            x = x * torch.empty(1).uniform_(0.75, 1.25).item()

        return x

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        """
        Returns:
            volume:  Tensor (1, 256, 256, 256)
            label:   Tensor scalar int64
            case_id: str
        """
        case_id = self.case_ids[idx]
        volume  = self._load_volume(case_id)
        label   = torch.tensor(self.labels[case_id], dtype=torch.long)

        if self.augment and self.split == "train":
            volume = self._augment(volume)

        return volume, label, case_id


class MeningiomaT1cDataModule:
    """
    Convenience wrapper that creates train/val/test DataLoaders.

    Note: 256³ volumes are large — default batch_size=1 to avoid OOM.
    """

    def __init__(
        self,
        data_root:   str,
        labels_path: str,
        splits_path: str,
        fold:        int  = 0,
        batch_size:  int  = 1,
        num_workers: int  = 4,
        augment:     bool = True,
    ):
        self.data_root   = data_root
        self.labels_path = labels_path
        self.splits_path = splits_path
        self.fold        = fold
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.augment     = augment

    def _make_dataset(self, split: str, augment: bool = False) -> MeningiomaT1cDataset:
        return MeningiomaT1cDataset(
            data_root   = self.data_root,
            labels_path = self.labels_path,
            splits_path = self.splits_path,
            split       = split,
            fold        = self.fold,
            augment     = augment,
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
    DATA_ROOT   = "/home/jma/Documents/projects/aaron/nnssl_dataset/nnssl_preprocessed/Dataset010_T1c_20260115/nnsslPlans_onemmiso/Dataset010_T1c_20260115/Dataset010_T1c_20260115"
    LABELS_PATH = "/home/jma/Documents/projects/aaron/nnssl_dataset/nnssl_preprocessed/Dataset010_T1c_20260115/labels.json"
    SPLITS_PATH = "/home/jma/Documents/projects/aaron/nnssl_dataset/nnssl_preprocessed/Dataset010_T1c_20260115/splits_hold_out.json"

    print("=" * 60)
    print("MeningiomaT1cDataset Sanity Check")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        print(f"\n--- {split} ---")
        ds = MeningiomaT1cDataset(
            data_root   = DATA_ROOT,
            labels_path = LABELS_PATH,
            splits_path = SPLITS_PATH,
            split       = split,
            augment     = (split == "train"),
        )
        vol, label, case_id = ds[0]
        print(f"  First case:    {case_id}")
        print(f"  Volume shape:  {vol.shape}")
        print(f"  Volume dtype:  {vol.dtype}")
        print(f"  Min/Max:       {vol.min():.3f} / {vol.max():.3f}")
        print(f"  Label:         {label.item()} (MG{label.item()+1})")

    print("\n--- DataLoader test ---")
    dm = MeningiomaT1cDataModule(
        data_root   = DATA_ROOT,
        labels_path = LABELS_PATH,
        splits_path = SPLITS_PATH,
        batch_size  = 1,
        num_workers = 2,
    )
    loader = dm.train_dataloader()
    vols, labels, ids = next(iter(loader))
    print(f"  Batch volumes: {vols.shape}")   # expect [1, 1, 256, 256, 256]
    print(f"  Batch labels:  {labels}")
    print(f"  Case IDs:      {list(ids)}")
    print("\nAll checks passed!")