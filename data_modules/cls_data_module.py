"""
Generic 3D Classification Dataset for AnyMC3D (v5 — single unified module)
===========================================================================
A single data module that covers PDCAD, BMLMPS, meningioma, and any future
3D volume classification dataset whose .npy files follow the convention
``{data_root}/{case_id}_0000.npy``.  All dataset-specific behaviour is
driven by the Hydra data config — no per-dataset Python subclass needed.

What differs per dataset, and how it's resolved
-----------------------------------------------
    1.  Label file format (.json vs .csv)
        → auto-detected from the file extension of ``labels_path``.
        → JSON must be a dict ``{case_id: int_label}``.
        → CSV must have header with ``label_col``, default ``label``,
          plus an id column, default ``identifier``.

    2.  Splits file format
        → auto-detected from the top-level JSON structure:
            - dict keyed by fold number: ``{"0": {"train": [...], "val": [...]}}``
            - list indexed by fold:      ``[{"train": [...], "val": [...]}, ...]``

    3.  Which splits exist (train/val/test vs train/val)
        → inferred at lookup time; ``test_dataloader()`` returns ``None`` when
          no test split exists in the selected fold.

What the config carries as metadata only (read by inference_nifti.py)
---------------------------------------------------------------------
    preprocess_strategy : 'percentile' | 'zscore' | 'ct_window' | 'none'
    class_names         : list, e.g. [0, 1] or ['NoCad', 'CAD']

These fields are accepted by __init__ so Hydra can pass them through from
the data config, but the DataModule itself does not use them — the
inference script reads them directly from the saved config.yaml.

Responsibilities
----------------
    ClassificationDataset    : loads and resizes a pre-normalized ``.npy``
                               volume.  Returns (volume, label, case_id)
                               with NO augmentation.
    ClassificationDataModule : builds train/val/(test) DataLoaders,
                               optionally wrapping the base dataset with a
                               ``TransformedDataset`` when transform
                               attributes are set.

Volume convention
-----------------
    Input  .npy: (1, H0, W0, S0) float32, already in [0, 1]
    Output:      (1, H,  W,  S ) float32, in [0, 1]  (per patch_size)

Sanity check
------------
    python -m data_modules.classification_dataset
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader, Dataset

from .data_augmentation import TransformedDataset


# ---------------------------------------------------------------
# Label loaders — dispatched by file extension
# ---------------------------------------------------------------

def _load_labels_json(labels_path: Path) -> dict:
    """
    Load a JSON label file of shape ``{case_id: int_label}``.
    """
    with open(labels_path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{labels_path}: expected a JSON object (dict), got {type(raw).__name__}"
        )
    return {str(k): int(v) for k, v in raw.items()}


def _load_labels_csv(
    labels_path: Path,
    id_col:      str = "identifier",
    label_col:   str = "label",
) -> dict:
    """
    Load a CSV label file with header row containing ``id_col`` and ``label_col``.
    """
    labels: dict = {}
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{labels_path}: CSV has no header row")
        if id_col not in reader.fieldnames or label_col not in reader.fieldnames:
            raise ValueError(
                f"{labels_path}: CSV must have columns '{id_col}' and "
                f"'{label_col}'; got {reader.fieldnames}"
            )
        for row in reader:
            ident = (row[id_col] or "").strip()
            if not ident:
                continue
            labels[ident] = int(row[label_col])
    return labels

def _load_labels_csv_multilabel(
    labels_path: Path,
    id_col:      str       = "VolumeName",
    label_cols:  List[str] = None,
    strip_ext:   bool      = True,
) -> dict:
    """
    Load a CSV multi-label file. Returns {case_id: [int, int, ...]} where
    each list has len(label_cols) entries.

    Args:
        id_col:     Column with case identifier (e.g. 'VolumeName').
        label_cols: List of column names — order is preserved and defines
                    the order of labels in the model output.
        strip_ext:  If True, strip '.nii.gz' / '.nii' from id values so
                    they match the .npy filename stems.
    """
    if label_cols is None:
        raise ValueError("label_cols must be provided for multi-label CSV")

    labels: dict = {}
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{labels_path}: CSV has no header row")

        missing = [c for c in [id_col] + label_cols if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"{labels_path}: missing columns {missing}; "
                f"available: {reader.fieldnames}"
            )

        for row in reader:
            ident = (row[id_col] or "").strip()
            if not ident:
                continue
            if strip_ext:
                for ext in (".nii.gz", ".nii"):
                    if ident.endswith(ext):
                        ident = ident[:-len(ext)]
                        break
            labels[ident] = [int(row[c]) for c in label_cols]
    return labels


def _load_labels(
    labels_path: Union[str, Path],
    task:        str       = "multiclass",   # NEW
    id_col:      str       = "identifier",
    label_col:   str       = "label",
    label_cols:  List[str] = None,           # NEW (multilabel only)
) -> dict:
    p = Path(labels_path)
    if not p.exists():
        raise FileNotFoundError(f"Label file not found: {p}")

    suffix = p.suffix.lower()
    if task == "multilabel":
        if suffix != ".csv":
            raise ValueError("Multi-label labels currently require a .csv file.")
        return _load_labels_csv_multilabel(
            p, id_col=id_col, label_cols=label_cols
        )

    # multiclass (existing behaviour)
    if suffix == ".json":
        return _load_labels_json(p)
    if suffix == ".csv":
        return _load_labels_csv(p, id_col=id_col, label_col=label_col)
    raise ValueError(f"Unsupported label file extension '{suffix}' for {p}.")


# ---------------------------------------------------------------
# Splits loader — dict-of-folds and list-of-folds both supported
# ---------------------------------------------------------------

def _load_fold_splits(splits_path: Union[str, Path], fold: int) -> dict:
    """
    Load a JSON splits file and return the split dict for a given fold.

    Supports two common layouts:

      1. Dict keyed by fold (PDCAD / nnU-Net style):
           {"0": {"train": [...], "val": [...], "test": [...]},
            "1": {...},
            ...}

      2. List indexed by fold (BMLMPS / nnSSL style):
           [{"train": [...], "val": [...]},
            {"train": [...], "val": [...]},
            ...]

    Returns:
        A dict like ``{"train": [...], "val": [...]}`` — and possibly
        ``"test"`` too, if present in the source file.
    """
    p = Path(splits_path)
    if not p.exists():
        raise FileNotFoundError(f"Splits file not found: {p}")

    with open(p) as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        # Keys might be strings ("0") or ints (0); try both.
        key_str = str(fold)
        if key_str in raw:
            fold_splits = raw[key_str]
        elif fold in raw:
            fold_splits = raw[fold]
        else:
            raise KeyError(
                f"Fold {fold} not found in {p}; "
                f"available fold keys: {list(raw.keys())}"
            )
    elif isinstance(raw, list):
        if not (0 <= fold < len(raw)):
            raise IndexError(
                f"Fold {fold} out of range for list-of-folds splits "
                f"(length {len(raw)}) in {p}"
            )
        fold_splits = raw[fold]
    else:
        raise ValueError(
            f"{p}: expected dict or list at top level, got {type(raw).__name__}"
        )

    if not isinstance(fold_splits, dict):
        raise ValueError(
            f"{p}: fold entry must be a dict of {{split_name: [case_ids]}}, "
            f"got {type(fold_splits).__name__}"
        )
    return fold_splits


# ---------------------------------------------------------------
# Generic dataset
# ---------------------------------------------------------------

class ClassificationDataset(Dataset):
    """
    Generic 3D classification dataset.

    Loads a pre-normalized ``.npy`` volume from
    ``{data_root}/{case_id}_0000.npy``, resizes it to ``patch_size`` via
    trilinear interpolation, and returns ``(volume, label, case_id)``.
    No MONAI transforms are applied here.

    Args:
        data_root:      Directory containing ``{case_id}_0000.npy`` files.
        labels_path:    Path to labels (.json dict or .csv table).
        splits_path:    Path to splits JSON (dict-of-folds or list-of-folds).
        split:          'train', 'val', or 'test'.
        fold:           Fold index.
        patch_size:     Target volume size [H, W, S] after trilinear resize.
        class_names:    Optional display names; purely cosmetic for logging.
        id_col:         CSV column holding case ids (default 'identifier').
        label_col:      CSV column holding int labels (default 'label').
        file_suffix:    Filename suffix between case_id and .npy
                        (default '_0000' for nnU-Net-style channel suffix).
        task:           'multiclass' or 'multilabel' (default 'multiclass').
        label_cols:     For 'multilabel' task, list of CSV columns to read
    """

    def __init__(
        self,
        data_root:    str,
        labels_path:  str,
        splits_path:  str,
        split:        str  = "train",
        fold:         int  = 0,
        patch_size:   list = None,
        class_names:  list = None,
        id_col:       str  = "identifier",
        label_col:    str  = "label",
        file_suffix:  str  = "_0000",
        task:         str  = "multiclass",
        label_cols:   list = None,
    ):
        self.data_root   = Path(data_root)
        self.split       = split
        self.patch_size  = patch_size or [308, 308, 70]
        self.class_names = class_names
        self.file_suffix = file_suffix
        self.task        = task
        self.label_cols  = label_cols

        # ── Labels (auto-dispatched by file extension) ───────────────────────
        self.labels = _load_labels(labels_path, task=task, id_col=id_col, label_col=label_col, label_cols=label_cols)

        # ── Splits (auto-dispatched by JSON structure) ───────────────────────
        fold_splits = _load_fold_splits(splits_path, fold)
        if split not in fold_splits:
            available = list(fold_splits.keys())
            raise KeyError(
                f"Fold {fold} has no '{split}' split; available: {available}"
            )
        self.case_ids = list(fold_splits[split])

        # ── Drop cases with no label ─────────────────────────────────────────
        missing_labels = [c for c in self.case_ids if c not in self.labels]
        if missing_labels:
            print(f"WARNING: {len(missing_labels)} cases in {split} split have no label, skipping:")
            for m in missing_labels[:10]:
                print(f"  {m}")
            self.case_ids = [c for c in self.case_ids if c in self.labels]

        # ── Drop cases with no .npy file on disk ─────────────────────────────
        missing_files = [c for c in self.case_ids if not self._get_npy_path(c).exists()]
        if missing_files:
            print(f"WARNING: {len(missing_files)} cases in {split} split have no .npy file, skipping:")
            for m in missing_files[:10]:
                print(f"  {self._get_npy_path(m)}")
            missing_set = set(missing_files)
            self.case_ids = [c for c in self.case_ids if c not in missing_set]

        print(f"ClassificationDataset [{split}, fold={fold}]: {len(self.case_ids)} cases")
        print(f"  patch_size: {self.patch_size}")
        self._print_class_distribution()

    # ------------------------------------------------------------------

    def _print_class_distribution(self):
        if self.task == "multilabel":
            n = len(self.case_ids)
            # Sum across cases per label
            counts = np.sum(
                [self.labels[c] for c in self.case_ids], axis=0
            ).astype(int)
            names = self.label_cols or [f"label_{i}" for i in range(len(counts))]
            print(f"  Multi-label positive rates ({n} cases):")
            for name, cnt in zip(names, counts):
                print(f"    {name:40s}  {cnt:5d}  ({100*cnt/n:.1f}%)")
            return

        counts = Counter(self.labels[c] for c in self.case_ids)
        if self.class_names:
            dist = {
                str(self.class_names[k]) if k < len(self.class_names) else f"Class{k}":
                    counts.get(k, 0)
                for k in sorted(counts)
            }
        else:
            dist = {f"Class{k}": counts.get(k, 0) for k in sorted(counts)}
        print(f"  Class distribution: {dist}")

    def _get_npy_path(self, case_id: str) -> Path:
        return self.data_root / f"{case_id}{self.file_suffix}.npy"

    def _load_volume(self, case_id: str) -> torch.Tensor:
        """
        Load pre-normalized volume and resize to patch_size.

        Input .npy:  (1, H0, W0, S0) float32, already in [0, 1]
        Output:      (1, H,  W,  S ) float32, in [0, 1]
        """
        path = self._get_npy_path(case_id)
        if not path.exists():
            raise FileNotFoundError(f"Volume not found: {path}")

        arr    = np.load(str(path))
        volume = torch.from_numpy(arr).float()

        target_H, target_W, target_S = self.patch_size
        _, cur_H, cur_W, cur_S = volume.shape

        if (cur_H != target_H) or (cur_W != target_W) or (cur_S != target_S):
            volume = volume.unsqueeze(0)  # (1, 1, H, W, S)
            volume = F.interpolate(
                volume,
                size=(target_H, target_W, target_S),
                mode="trilinear",
                align_corners=False,
            )
            volume = volume.squeeze(0)
            volume = volume.clamp(0.0, 1.0)

        return volume

    def __len__(self) -> int:
        return len(self.case_ids)

    def __getitem__(self, idx: int):
        case_id = self.case_ids[idx]
        volume  = self._load_volume(case_id)

        if self.task == "multilabel":
            # self.labels[case_id] is a list of 0/1 ints; convert to float tensor
            label = torch.tensor(self.labels[case_id], dtype=torch.float32)
        else:
            label = torch.tensor(self.labels[case_id], dtype=torch.long)

        return volume, label, case_id


# ---------------------------------------------------------------
# Generic data module
# ---------------------------------------------------------------

class ClassificationDataModule(L.LightningDataModule):
    """
    Generic data module covering any dataset that fits the
    ``ClassificationDataset`` contract.  Selected in Hydra with e.g.:

        module:
          _target_: data_modules.classification_dataset.ClassificationDataModule
          data_root:   /path/to/preprocessed
          labels_path: /path/to/labels.{json,csv}
          splits_path: /path/to/splits.json
          fold:        0
          batch_size:  2
          num_workers: 8
          patch_size:  [308, 308, 70]
          preprocess_strategy: percentile   # metadata for inference
          class_names: [0, 1]               # metadata for inference
          id_col:      identifier           # CSV only, default shown
          label_col:   label                # CSV only, default shown
          file_suffix: _0000                # default shown

    Augmentation is decoupled: ``train_transform`` / ``eval_transform``
    attributes are set externally (by ``apply_augmentation`` in train.py).

    When the splits file does not contain a 'test' entry for a given fold,
    ``test_dataloader()`` returns ``None`` so ``trainer.test(...)`` calls
    can be skipped by the caller.
    """

    def __init__(
        self,
        data_root:           str,
        labels_path:         str,
        splits_path:         str,
        fold:                int             = 0,
        batch_size:          int             = 2,
        num_workers:         int             = 8,
        patch_size:          list            = None,
        # ── Metadata — consumed by inference, accepted here as pass-through ──
        preprocess_strategy: Optional[str]   = None,
        class_names:         Optional[list]  = None,
        # ── Label file options (ignored for JSON labels) ─────────────────────
        id_col:              str             = "identifier",
        label_col:           str             = "label",
        # ── Filename convention ──────────────────────────────────────────────
        file_suffix:         str             = "_0000",
        task:                str             = "multiclass",
        label_cols:          Optional[list]  = None,
    ):
        super().__init__()
        self.data_root           = data_root
        self.labels_path         = labels_path
        self.splits_path         = splits_path
        self.fold                = fold
        self.batch_size          = batch_size
        self.num_workers         = num_workers
        self.patch_size          = patch_size or [308, 308, 70]
        self.preprocess_strategy = preprocess_strategy
        self.class_names         = class_names
        self.id_col              = id_col
        self.label_col           = label_col
        self.file_suffix         = file_suffix
        self.task                = task
        self.label_cols          = label_cols

        # Transform hooks — set by apply_augmentation() in train.py.
        self.train_transform = None
        self.eval_transform  = None

        # Cache the fold's available split names so we can silently skip
        # test_dataloader() when no 'test' split exists (e.g. BMLMPS).
        self._available_splits = self._peek_available_splits()

    # ------------------------------------------------------------------

    def _peek_available_splits(self) -> set:
        """Cheaply read the splits file to learn which split names exist."""
        try:
            fold_splits = _load_fold_splits(self.splits_path, self.fold)
            return set(fold_splits.keys())
        except Exception as e:
            print(f"WARNING: could not pre-read splits file: {e}")
            return {"train", "val"}

    def _make_dataset(self, split: str) -> ClassificationDataset:
        return ClassificationDataset(
            data_root   = self.data_root,
            labels_path = self.labels_path,
            splits_path = self.splits_path,
            split       = split,
            fold        = self.fold,
            patch_size  = self.patch_size,
            class_names = self.class_names,
            id_col      = self.id_col,
            label_col   = self.label_col,
            file_suffix = self.file_suffix,
            task        = self.task,
            label_cols  = self.label_cols,
        )

    # ------------------------------------------------------------------

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

    def test_dataloader(self) -> Optional[DataLoader]:
        if "test" not in self._available_splits:
            print("NOTE: no 'test' split in splits file — skipping test_dataloader.")
            return None
        ds = self._make_dataset("test")
        ds = TransformedDataset(ds, self.eval_transform)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


# ---------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------

if __name__ == "__main__":
    # Run from project root with:
    #     python -m data_modules.classification_dataset
    from .data_augmentation import apply_augmentation

    # PDCAD-style (JSON labels, dict-of-folds splits, has test split)
    DATA_ROOT   = "/home/jma/Documents/projects/aaron/AnyMC3D/preprocessed_data/PDCAD"
    LABELS_PATH = "/home/jma/Documents/projects/aaron/nnssl_dataset/nnssl_preprocessed/Dataset009_PDCAD_NM/labels.json"
    SPLITS_PATH = "/home/jma/Documents/projects/aaron/nnssl_dataset/nnssl_preprocessed/Dataset009_PDCAD_NM/splits.json"

    print("=" * 60)
    print("ClassificationDataModule sanity check")
    print("=" * 60)

    dm = ClassificationDataModule(
        data_root           = DATA_ROOT,
        labels_path         = LABELS_PATH,
        splits_path         = SPLITS_PATH,
        fold                = 0,
        batch_size          = 2,
        num_workers         = 2,
        patch_size          = [308, 308, 70],
        preprocess_strategy = "percentile",
        class_names         = [0, 1],
    )
    apply_augmentation(dm, augment_train=True)

    loader = dm.train_dataloader()
    vols, labels, ids = next(iter(loader))
    print(f"  Batch volumes: {vols.shape}")
    print(f"  Batch labels:  {labels}")
    print(f"  Case IDs:      {list(ids)}")
    print(f"  Available splits for fold 0: {sorted(dm._available_splits)}")
    print("\nAll checks passed!")