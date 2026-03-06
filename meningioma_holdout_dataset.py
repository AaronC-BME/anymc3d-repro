"""
Meningioma Dataset for AnyMC3D
Loads T1c cropped volumes from Blosc2 format with MG1-MG4 labels.

Data format:
    - Volumes: (1, 64, 64, 64) float32, already normalized
    - Labels:  0=MG1, 1=MG2, 2=MG3, 3=MG4
    - Splits:  train/val/test defined in splits_hold_out.json
"""

import json
import torch
import numpy as np
import blosc2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class MeningiomaDataset(Dataset):
    """
    PyTorch Dataset for meningioma molecular subtype classification.

    Args:
        data_root:   Path to the dataset root containing per-case folders
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
        split:       str = "train",
        fold:        int = 0,
        augment:     bool = False,
    ):
        assert split in ("train", "val", "test"), \
            f"split must be 'train', 'val', or 'test', got '{split}'"

        self.data_root = Path(data_root)
        self.split     = split
        self.augment   = augment

        # Load labels: {filename -> int label}
        with open(labels_path) as f:
            self.labels = json.load(f)

        # Load splits
        with open(splits_path) as f:
            splits = json.load(f)

        self.case_ids = splits[str(fold)][split]

        # Filter to cases that have labels
        missing = [c for c in self.case_ids if c not in self.labels]
        if missing:
            print(f"WARNING: {len(missing)} cases in {split} split have no label, skipping:")
            for m in missing:
                print(f"  {m}")
        self.case_ids = [c for c in self.case_ids if c in self.labels]

        print(f"MeningiomaDataset [{split}]: {len(self.case_ids)} cases")
        self._print_class_distribution()

    def _print_class_distribution(self):
        from collections import Counter
        counts = Counter(self.labels[c] for c in self.case_ids)
        class_names = {0: "MG1", 1: "MG2", 2: "MG3", 3: "MG4"}
        dist = {class_names[k]: counts.get(k, 0) for k in sorted(class_names)}
        print(f"  Class distribution: {dist}")

    def _get_b2nd_path(self, case_id: str) -> Path:
        """
        Structure: data_root/{case_id}.nii.gz/ses-DEFAULT/{stem}.b2nd
        """
        stem = case_id.replace(".nii.gz", "")
        return self.data_root / f"{case_id}" / "ses-DEFAULT" / f"{stem}.b2nd"
    
    def _pad_to_64(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pad volume to (1, 64, 64, 64) if any dimension is smaller.
        Pads symmetrically with zeros (background).
        x: (1, D, H, W)
        """
        _, d, h, w = x.shape
        target = 64
        pad_d = max(0, target - d)
        pad_h = max(0, target - h)
        pad_w = max(0, target - w)

        # F.pad takes padding in reverse order: (left, right, top, bottom, front, back)
        x = F.pad(x, (
            pad_w // 2, pad_w - pad_w // 2,   # W
            pad_h // 2, pad_h - pad_h // 2,   # H
            pad_d // 2, pad_d - pad_d // 2,   # D
        ))
        return x

    def _load_volume(self, case_id: str) -> torch.Tensor:
        path = self._get_b2nd_path(case_id)
        if not path.exists():
            raise FileNotFoundError(f"Volume not found: {path}")
        arr = blosc2.open(str(path))[:]
        arr = np.ascontiguousarray(arr)
        volume = torch.from_numpy(arr).float().clone()
        volume = self._pad_to_64(volume)   # <-- add this line
        return volume

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple augmentation matching the nnU-Net strategy from the AnyMC3D paper.
        Args:
            x: Tensor (1, 64, 64, 64)
        """
        # Random flips along all 3 spatial axes
        for dim in [1, 2, 3]:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[dim])

        # Random Gaussian noise
        if torch.rand(1).item() < 0.25:
            x = x + torch.randn_like(x) * 0.05

        # Random brightness scaling
        if torch.rand(1).item() < 0.15:
            x = x * torch.empty(1).uniform_(0.75, 1.25).item()

        return x

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        """
        Returns:
            volume:  Tensor (1, 64, 64, 64)
            label:   Tensor scalar int64
            case_id: String (useful for per-case analysis)
        """
        case_id = self.case_ids[idx]
        volume  = self._load_volume(case_id)
        label   = torch.tensor(self.labels[case_id], dtype=torch.long)

        if self.augment and self.split == "train":
            volume = self._augment(volume)

        return volume, label, case_id


class MeningiomaDataModule:
    """
    Convenience wrapper that creates train/val/test DataLoaders.

    Usage:
        dm = MeningiomaDataModule(data_root=..., labels_path=..., splits_path=...)
        train_loader = dm.train_dataloader()
        val_loader   = dm.val_dataloader()
        test_loader  = dm.test_dataloader()
    """

    def __init__(
        self,
        data_root:    str,
        labels_path:  str,
        splits_path:  str,
        fold:         int = 0,
        batch_size:   int = 4,
        num_workers:  int = 4,
        augment:      bool = True,
    ):
        self.data_root   = data_root
        self.labels_path = labels_path
        self.splits_path = splits_path
        self.fold        = fold
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.augment     = augment

    def _make_dataset(self, split: str, augment: bool = False) -> MeningiomaDataset:
        return MeningiomaDataset(
            data_root=self.data_root,
            labels_path=self.labels_path,
            splits_path=self.splits_path,
            split=split,
            fold=self.fold,
            augment=augment,
        )

    def train_dataloader(self) -> DataLoader:
        ds = self._make_dataset("train", augment=self.augment)
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True, drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        ds = self._make_dataset("val", augment=False)
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        ds = self._make_dataset("test", augment=False)
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
        )


if __name__ == "__main__":
    DATA_ROOT   = "/home/jma/Documents/projects/aaron/nnssl_dataset/nnssl_preprocessed/Dataset006_T1c_cropped_20260115/nnsslPlans_onemmiso/Dataset006_T1c_cropped_20260115/Dataset006_T1c_cropped_20260115"
    LABELS_PATH = "/home/jma/Documents/projects/aaron/nnssl_dataset/nnssl_preprocessed/Dataset006_T1c_cropped_20260115/labels.json"
    SPLITS_PATH = "/home/jma/Documents/projects/aaron/nnssl_dataset/nnssl_preprocessed/Dataset006_T1c_cropped_20260115/splits_hold_out.json"

    print("=" * 60)
    print("MeningiomaDataset Sanity Check")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        print(f"\n--- {split} ---")
        ds = MeningiomaDataset(
            data_root=DATA_ROOT,
            labels_path=LABELS_PATH,
            splits_path=SPLITS_PATH,
            split=split,
            augment=(split == "train"),
        )
        vol, label, case_id = ds[0]
        print(f"  First case:    {case_id}")
        print(f"  Volume shape:  {vol.shape}")
        print(f"  Volume dtype:  {vol.dtype}")
        print(f"  Min/Max:       {vol.min():.3f} / {vol.max():.3f}")
        print(f"  Label:         {label.item()} (MG{label.item()+1})")

    print("\n--- DataLoader test ---")
    dm = MeningiomaDataModule(
        data_root=DATA_ROOT,
        labels_path=LABELS_PATH,
        splits_path=SPLITS_PATH,
        batch_size=4,
        num_workers=2,
    )
    loader = dm.train_dataloader()
    vols, labels, ids = next(iter(loader))
    print(f"  Batch volumes: {vols.shape}")  # expect [4, 1, 64, 64, 64]
    print(f"  Batch labels:  {labels}")
    print(f"  Case IDs:      {list(ids)}")
    print("\nAll checks passed!")