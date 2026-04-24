"""
inference_nifti.py — AnyMC3D inference on raw NIfTI files
==========================================================
Applies the **full nnU-Net-style preprocessing pipeline** used at training
time (load → crop → normalize → resample → optional resize), not just
in-place normalization.  All preprocessing parameters are resolved in this
priority order:

    1.  CLI flag (explicit override)
    2.  preprocessing_manifest.json inside --run_dir
        (copied there automatically by train.py at training start)
    3.  cfg.data fields in the saved config.yaml
    4.  Safe defaults

This guarantees the volume fed to the model at inference is identical in
construction to what the model saw during training.

────────────────────────────────────────────────────────────────────────────────
USAGE — default (manifest read from run_dir, no extra flags needed)
────────────────────────────────────────────────────────────────────────────────
  python inference_nifti.py \\
      --run_dir   checkpoints/anymc3d-pdcad-fold0 \\
      --nifti_dir /data/PDCAD/raw_nifti

────────────────────────────────────────────────────────────────────────────────
USAGE — with ground-truth labels
────────────────────────────────────────────────────────────────────────────────
  python inference_nifti.py \\
      --run_dir    checkpoints/vjepa21-vitb-pdcad-fold0 \\
      --nifti_dir  /data/PDCAD/raw_nifti \\
      --labels_csv /data/PDCAD/val_cases.csv \\
      --checkpoint "epoch=50-val_auroc=0.8808.ckpt"

────────────────────────────────────────────────────────────────────────────────
USAGE — manual override (legacy runs without a manifest in run_dir)
────────────────────────────────────────────────────────────────────────────────
  python inference_nifti.py \\
      --run_dir        checkpoints/anymc3d-pdcad-fold0 \\
      --nifti_dir      /data/PDCAD/raw_nifti \\
      --norm           percentile \\
      --lower_pct      0 --upper_pct 99.5 \\
      --target_spacing 1.0,1.0,1.0 \\
      --crop_margin    4

────────────────────────────────────────────────────────────────────────────────
PREPROCESSING PIPELINE (matches preprocess.py v4)
────────────────────────────────────────────────────────────────────────────────
  1. Load .nii.gz, reorient to RAS+, read voxel spacing
  2. Crop to nonzero bounding box (margin voxels on each side)
  3. Normalize:
       zscore      — (x - μ)/σ on foreground voxels only
       percentile  — ScaleIntensityRangePercentiles(lower, upper) → [0, 1]
       none        — pass-through (if volumes are already preprocessed)
  4. Resample to target_spacing via scipy.ndimage.zoom (trilinear)
  5. Add channel dim → (1, H, W, S)
  6. Optional resize to target_size³ (trilinear)

The exact order mirrors nnU-Net: crop before normalize (so foreground stats
aren't diluted by background), normalize before resample (so the nonzero
mask aligns perfectly with the image).

────────────────────────────────────────────────────────────────────────────────
NIFTI LAYOUT
────────────────────────────────────────────────────────────────────────────────
  Flat:         nifti_dir/RJPD_000_0000.nii.gz
  Subdir:       nifti_dir/RJPD_000/RJPD_000_0000.nii.gz

────────────────────────────────────────────────────────────────────────────────
OUTPUTS (saved inside run_dir)
────────────────────────────────────────────────────────────────────────────────
  predictions_nifti.csv          — per-case predictions + probabilities
  summary_nifti.csv              — overall + per-class metrics   (labels only)
  confusion_matrix_nifti.png     — confusion matrix               (labels only)
  roc_curves_nifti.png           — per-class ROC curves           (labels only)

python inference_nifti_fullPreproc.py \
      --run_dir   checkpoints/vjepa21-vitb-pdcad_fullPreproc_num_frames64_loraLR_1e-4_headLR_1e-3_DA_fold0 \
      --nifti_dir /home/jma/Documents/projects/safwat/Datasets/Dataset031_PDCAD_NM/imagesVal \
      --labels_csv /home/jma/Documents/projects/safwat/Datasets/Dataset031_PDCAD_NM/val_cases.csv \
      --checkpoint "epoch=50-val_auroc=0.8808.ckpt" 
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score, roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

CLASS_NAMES: list[str] = []   # set in main()


# ============================================================================
#  Small helpers
# ============================================================================

def _strip_channel_suffix(s: str) -> str:
    """'RJPD_000_0000' → 'RJPD_000'; 'RJPD_000' → 'RJPD_000'."""
    return s[:-5] if s.endswith("_0000") else s


def _parse_spacing(s: str) -> tuple[float, float, float]:
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"spacing must be 'H,W,S', got {s!r}")
    return tuple(parts)  # type: ignore[return-value]


# ============================================================================
#  Preprocessing configuration
# ============================================================================

class PreprocConfig:
    """
    Bundle of all parameters for the full preprocessing pipeline.

    Built by merging, in priority order:
      1. CLI overrides
      2. preprocessing_manifest.json in --manifest_dir
      3. cfg.data fields from config.yaml
      4. Safe defaults
    """

    def __init__(
        self,
        norm:            str                             = "zscore",
        lower_pct:       float                           = 0.5,
        upper_pct:       float                           = 99.5,
        target_spacing:  tuple[float, float, float] | None = None,
        crop_margin:     int                             = 4,
        target_size:     int | None                      = None,
        do_crop:         bool                            = True,
        do_resample:     bool                            = True,
    ):
        self.norm            = norm
        self.lower_pct       = lower_pct
        self.upper_pct       = upper_pct
        self.target_spacing  = target_spacing
        self.crop_margin     = crop_margin
        self.target_size     = target_size
        self.do_crop         = do_crop
        self.do_resample     = do_resample

    def log_summary(self) -> None:
        log.info("Preprocessing pipeline:")
        log.info(f"  crop          : {'ON  (margin=' + str(self.crop_margin) + ' vox)' if self.do_crop else 'OFF'}")
        if self.norm == "percentile":
            log.info(f"  normalize     : percentile clip [{self.lower_pct}, {self.upper_pct}] → [0, 1]")
        elif self.norm == "zscore":
            log.info(f"  normalize     : z-score (foreground voxels only)")
        elif self.norm == "none":
            log.info(f"  normalize     : OFF (pass-through)")
        else:
            log.info(f"  normalize     : {self.norm}")
        if self.do_resample and self.target_spacing is not None:
            sp = self.target_spacing
            log.info(f"  resample to   : {sp[0]:.4f} × {sp[1]:.4f} × {sp[2]:.4f} mm")
        else:
            log.info(f"  resample      : OFF")
        if self.target_size is not None:
            log.info(f"  final resize  : {self.target_size}³ (trilinear)")
        else:
            log.info(f"  final resize  : OFF")


# ============================================================================
#  Manifest / config resolution
# ============================================================================

def _load_manifest(run_dir: Path) -> dict:
    """
    Read preprocessing_manifest.json from the run directory (copied there by
    train.py at the start of training).  Returns an empty dict if the file
    does not exist — the script then falls back to cfg.data fields and
    safe defaults.
    """
    p = run_dir / "preprocessing_manifest.json"
    if not p.exists():
        log.warning(
            f"No preprocessing_manifest.json in {run_dir}. Falling back to "
            f"cfg.data fields + defaults. If you trained before the manifest "
            f"was auto-copied, pass --norm/--target_spacing/etc. explicitly."
        )
        return {}
    with open(p) as f:
        m = json.load(f)
    log.info(f"Loaded preprocessing manifest from {p}")
    return m


def build_preproc_config(cfg, manifest: dict, args) -> PreprocConfig:
    """
    Resolve every preprocessing parameter with the priority chain:
      CLI > manifest > cfg.data > default
    """
    data_cfg = cfg.get("data", {}) if cfg is not None else {}

    # ── Normalization strategy ───────────────────────────────────────────────
    if args.norm:
        norm = args.norm
    elif manifest.get("normalization"):
        norm = manifest["normalization"]
    elif "preprocess_strategy" in data_cfg:
        # Translate the inference-script vocabulary to preprocess.py vocabulary.
        strat = data_cfg.preprocess_strategy
        norm = {"percentile": "percentile",
                "zscore":     "zscore",
                "none":       "none",
                "ct_window":  "percentile"}.get(strat, strat)
    else:
        norm = _infer_norm_from_dataset(data_cfg.get("dataset", ""))

    # ── Percentile bounds ────────────────────────────────────────────────────
    lower_pct = (args.lower_pct if args.lower_pct is not None
                 else manifest.get("percentile_lower")
                 if manifest.get("percentile_lower") is not None else 0.5)
    upper_pct = (args.upper_pct if args.upper_pct is not None
                 else manifest.get("percentile_upper")
                 if manifest.get("percentile_upper") is not None else 99.5)

    # ── Target spacing ───────────────────────────────────────────────────────
    if args.target_spacing:
        target_spacing = _parse_spacing(args.target_spacing)
    elif manifest.get("target_spacing_mm"):
        target_spacing = tuple(manifest["target_spacing_mm"])  # type: ignore[assignment]
    else:
        target_spacing = None

    # ── Crop margin ──────────────────────────────────────────────────────────
    if args.crop_margin is not None:
        crop_margin = args.crop_margin
    elif manifest.get("crop_margin_vox") is not None:
        crop_margin = int(manifest["crop_margin_vox"])
    else:
        crop_margin = 4

    # ── Optional final resize ────────────────────────────────────────────────
    if args.target_size is not None:
        target_size = args.target_size
    elif manifest.get("target_size") is not None:
        target_size = int(manifest["target_size"])
    else:
        target_size = None

    # ── Flags to disable individual steps ────────────────────────────────────
    do_crop     = (not args.no_crop)     and crop_margin is not None
    do_resample = (not args.no_resample) and (target_spacing is not None)

    return PreprocConfig(
        norm            = norm,
        lower_pct       = float(lower_pct),
        upper_pct       = float(upper_pct),
        target_spacing  = target_spacing,
        crop_margin     = int(crop_margin),
        target_size     = target_size,
        do_crop         = do_crop,
        do_resample     = do_resample,
    )


def _infer_norm_from_dataset(dataset_name: str) -> str:
    mapping = {"pdcad": "percentile", "bmlmps": "percentile", "meningioma": "zscore"}
    return mapping.get(dataset_name.lower(), "percentile")


# ============================================================================
#  Preprocessing steps (match preprocess.py v4 exactly)
# ============================================================================

def load_nifti_with_spacing(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """
    Load NIfTI, reorient to RAS+ canonical, return (data, spacing).
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required: pip install nibabel")
    img           = nib.load(str(path))
    img_canonical = nib.as_closest_canonical(img)
    spacing       = tuple(float(s) for s in img_canonical.header.get_zooms()[:3])
    data          = img_canonical.get_fdata(dtype=np.float32)
    while data.ndim > 3:
        data = data.squeeze(-1)
    return data, spacing  # type: ignore[return-value]


def crop_to_nonzero(data: np.ndarray, margin: int = 4) -> np.ndarray:
    """Crop to bounding box of nonzero voxels, with margin clamped to image."""
    nz = np.argwhere(data != 0)
    if len(nz) == 0:
        return data
    lo = np.maximum(nz.min(axis=0) - margin, 0)
    hi = np.minimum(nz.max(axis=0) + margin + 1, np.array(data.shape))
    return data[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]


def zscore_normalize(data: np.ndarray) -> np.ndarray:
    """Foreground-masked z-score (matches MONAI NormalizeIntensity nonzero=True)."""
    mask = data != 0
    if mask.sum() < 100:
        return np.zeros_like(data, dtype=np.float32)
    mu  = float(data[mask].mean())
    sig = float(data[mask].std())
    out = np.zeros_like(data, dtype=np.float32)
    if sig > 1e-6:
        out[mask] = (data[mask] - mu) / sig
    return out


def percentile_normalize(
    data:      np.ndarray,
    lower_pct: float = 0.5,
    upper_pct: float = 99.5,
) -> np.ndarray:
    """ScaleIntensityRangePercentiles → [0, 1] (matches MONAI transform)."""
    lo = float(np.percentile(data, lower_pct))
    hi = float(np.percentile(data, upper_pct))
    if hi - lo < 1e-8:
        return np.zeros_like(data, dtype=np.float32)
    out = (data - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def resample_to_spacing(
    data:        np.ndarray,
    src_spacing: tuple[float, float, float],
    tgt_spacing: tuple[float, float, float],
) -> np.ndarray:
    """Trilinear resample via scipy.ndimage.zoom (order=1)."""
    from scipy.ndimage import zoom
    zoom_factors = tuple(s / t for s, t in zip(src_spacing, tgt_spacing))
    if np.allclose(zoom_factors, 1.0, atol=1e-3):
        return data.astype(np.float32)
    return zoom(data, zoom_factors, order=1).astype(np.float32)


def resize_volume(data: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize (1, H, W, S) → (1, T, T, T) via torch trilinear interpolation.
    """
    t = torch.from_numpy(data).unsqueeze(0)   # (1, 1, H, W, S)
    t = F.interpolate(
        t,
        size=(target_size, target_size, target_size),
        mode="trilinear",
        align_corners=False,
    )
    return t.squeeze(0).numpy().astype(np.float32)


def preprocess_full(path: Path, pc: PreprocConfig) -> np.ndarray:
    """
    Apply the full nnU-Net-style pipeline to a single NIfTI file.

    Pipeline order (matches preprocess.py v4):
        load → crop → normalize → resample → add channel → optional resize
    """
    # 1. Load + spacing
    data, src_spacing = load_nifti_with_spacing(path)

    # 2. Crop to nonzero bounding box (before normalize)
    if pc.do_crop:
        data = crop_to_nonzero(data, margin=pc.crop_margin)

    # 3. Normalize (before resample — foreground mask must align with image)
    if pc.norm == "zscore":
        data = zscore_normalize(data)
    elif pc.norm == "percentile":
        data = percentile_normalize(data, pc.lower_pct, pc.upper_pct)
    elif pc.norm == "none":
        data = data.astype(np.float32)
    else:
        raise ValueError(f"Unknown normalization: {pc.norm!r}")

    # 4. Resample to target spacing
    if pc.do_resample and pc.target_spacing is not None:
        data = resample_to_spacing(data, src_spacing, pc.target_spacing)

    # 5. Add channel dim → (1, H, W, S)
    data = data[np.newaxis, ...]

    # 6. Optional resize to target_size³
    if pc.target_size is not None:
        data = resize_volume(data, pc.target_size)

    return data.astype(np.float32)


# ============================================================================
#  Config resolution for model / class names
# ============================================================================

def resolve_num_classes(cfg) -> int:
    try:
        return int(cfg.model.num_classes)
    except (AttributeError, KeyError):
        raise ValueError("cfg.model.num_classes not found in config.yaml")


def resolve_class_names(cfg, cli_override: list[str] | None, num_classes: int) -> list[str]:
    if cli_override:
        if len(cli_override) != num_classes:
            raise ValueError(
                f"--class_names has {len(cli_override)} entries but "
                f"cfg.model.num_classes = {num_classes}"
            )
        return cli_override

    data_cfg = cfg.get("data", {})
    if "class_names" in data_cfg:
        names = [str(n) for n in data_cfg.class_names]
        if len(names) == num_classes:
            return names
        log.warning(
            f"cfg.data.class_names has {len(names)} entries but "
            f"num_classes = {num_classes}. Falling back to generic names."
        )
    return [f"Class{i}" for i in range(num_classes)]


# ============================================================================
#  NIfTI discovery
# ============================================================================

def find_config(run_dir: Path) -> Path:
    c = run_dir / "config.yaml"
    if c.exists():
        return c
    raise FileNotFoundError(f"config.yaml not found in {run_dir}")


def find_best_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("*.ckpt"))
    if not ckpts:
        ckpts = sorted(run_dir.rglob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {run_dir}")

    scored: list[tuple[float, Path]] = []
    for ckpt in ckpts:
        try:
            auroc_str = ckpt.stem.split("val_auroc=")[1]
            scored.append((float(auroc_str), ckpt))
        except (IndexError, ValueError):
            pass

    if scored:
        score, best = max(scored, key=lambda t: t[0])
        log.info(f"Auto-selected checkpoint (val_auroc={score:.4f}): {best.name}")
    else:
        best = max(ckpts, key=lambda p: p.stat().st_mtime)
        log.info(f"Auto-selected most-recent checkpoint: {best.name}")
    return best


def find_nifti_files(nifti_dir: Path) -> list[tuple[str, Path]]:
    """Discover NIfTI files in flat or one-level-subdirectory layouts."""
    hits: list[tuple[str, Path]] = []

    for ext in ("*.nii.gz", "*.nii"):
        for p in sorted(nifti_dir.glob(ext)):
            stem    = p.name.replace(".nii.gz", "").replace(".nii", "")
            case_id = _strip_channel_suffix(stem)
            hits.append((case_id, p))

    if not hits:
        for subdir in sorted(nifti_dir.iterdir()):
            if not subdir.is_dir():
                continue
            for ext in ("*.nii.gz", "*.nii"):
                for p in sorted(subdir.glob(ext)):
                    case_id = _strip_channel_suffix(subdir.name)
                    hits.append((case_id, p))

    if not hits:
        raise FileNotFoundError(
            f"No .nii / .nii.gz files found in {nifti_dir} "
            f"(checked flat + one-level subdirectories)"
        )

    seen: set[str] = set()
    unique: list[tuple[str, Path]] = []
    for case_id, p in hits:
        if case_id not in seen:
            seen.add(case_id)
            unique.append((case_id, p))

    log.info(f"Found {len(unique)} NIfTI file(s) in {nifti_dir}")
    return unique


# ============================================================================
#  Dataset
# ============================================================================

class NiftiInferenceDataset(Dataset):
    """
    Loads NIfTI files and runs the full preprocessing pipeline on the fly.

    Returns:
        volume  : torch.Tensor (1, H, W, S) float32
        label   : int   (-1 if unknown)
        case_id : str
    """

    def __init__(
        self,
        cases:     list[tuple[str, Path]],
        pc:        PreprocConfig,
        label_map: dict[str, int] | None = None,
    ):
        self.cases     = cases
        self.pc        = pc
        self.label_map = label_map or {}

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int):
        case_id, path = self.cases[idx]
        label = self.label_map.get(case_id, -1)
        vol   = preprocess_full(path, self.pc)   # (1, H, W, S)
        return torch.from_numpy(vol), label, case_id


# ============================================================================
#  Model loading + inference
# ============================================================================

def _resolve_class_from_target(target: str):
    import importlib
    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_model(ckpt_path: Path, cfg):
    """
    Load a Lightning checkpoint.  V-JEPA base weights are skipped because the
    Lightning .ckpt already contains the full model state.
    """
    target = cfg.model._target_
    log.info(f"Resolving model class: {target}")
    LightningClass = _resolve_class_from_target(target)

    if "vjepa2" in target.lower():
        model = LightningClass.load_from_checkpoint(
            str(ckpt_path), map_location="cpu", vjepa_checkpoint_path=None,
        )
    else:
        model = LightningClass.load_from_checkpoint(
            str(ckpt_path), map_location="cpu",
        )
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model,
    dataloader: DataLoader,
    device:     torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()

    uses_modality_dict = hasattr(model, "modalities") and model.modalities

    all_case_ids: list[str]        = []
    all_labels:   list[int]        = []
    all_probs:    list[np.ndarray] = []

    for volumes, labels, case_ids in dataloader:
        volumes = volumes.to(device)
        if uses_modality_dict:
            modality_key = model.modalities[0]
            logits, _ = model.model({modality_key: volumes})
        else:
            logits = model.model(volumes)

        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_case_ids.extend(list(case_ids))
        all_labels.extend(labels.numpy().tolist())
        all_probs.append(probs)

    return (
        all_case_ids,
        np.array(all_labels, dtype=np.int64),
        np.concatenate(all_probs, axis=0),
    )


@torch.no_grad()
def run_inference_variable_shape(
    model,
    dataset:      Dataset,
    device:       torch.device,
    num_workers:  int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Fallback inference path that handles volumes of different shapes by
    running one case at a time.  Used when target_size is not set and the
    crop / resample output shape varies per case.
    """
    model.to(device).eval()
    uses_modality_dict = hasattr(model, "modalities") and model.modalities

    loader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    all_case_ids: list[str]        = []
    all_labels:   list[int]        = []
    all_probs:    list[np.ndarray] = []

    for vol, label, case_ids in loader:
        vol = vol.to(device)
        if uses_modality_dict:
            key = model.modalities[0]
            logits, _ = model.model({key: vol})
        else:
            logits = model.model(vol)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_case_ids.extend(list(case_ids))
        all_labels.extend(label.numpy().tolist())
        all_probs.append(probs)

    return (
        all_case_ids,
        np.array(all_labels, dtype=np.int64),
        np.concatenate(all_probs, axis=0),
    )


# ============================================================================
#  Metrics + plots
# ============================================================================

def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """
    Find the binary decision threshold that maximizes Youden's J statistic
    (J = sensitivity + specificity - 1 = TPR - FPR).

    This is the point on the ROC curve farthest from the diagonal — the
    standard choice for "optimal" threshold when TPR and FPR are weighted
    equally.  Alternatives include F1-maximizing or cost-weighted thresholds;
    switch here if you need a different criterion.

    Args:
        y_true  : (N,) int array of ground-truth labels in {0, 1}
        y_score : (N,) float array of predicted P(class = 1)

    Returns:
        (threshold, youden_j)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    return float(thresholds[best_idx]), float(j_scores[best_idx])


def compute_balanced_accuracy(y_true, y_pred, num_classes: int) -> float:
    bal_accs = []
    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        tn = int(((y_pred != c) & (y_true != c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_accs.append((recall + specificity) / 2)
    return float(np.mean(bal_accs))


def compute_auroc(y_true, probs, num_classes: int):
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if num_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    per_class: list[float] = []
    roc_data:  list        = []
    for c in range(num_classes):
        if y_bin[:, c].sum() == 0:
            per_class.append(float('nan'))
            roc_data.append((None, None, CLASS_NAMES[c]))
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, c], probs[:, c])
        per_class.append(auc(fpr, tpr))
        roc_data.append((fpr, tpr, CLASS_NAMES[c]))

    valid = [v for v in per_class if not np.isnan(v)]
    macro = float(np.mean(valid)) if valid else float('nan')
    return per_class, macro, roc_data


def compute_per_class_stats(y_true, y_pred, per_class_auroc, num_classes: int):
    rows = []
    for c in range(num_classes):
        if (y_true == c).sum() == 0:
            continue
        tp = int(((y_pred == c) & (y_true == c)).sum())
        tn = int(((y_pred != c) & (y_true != c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1          = (2 * precision * recall / (precision + recall)
                       if (precision + recall) > 0 else 0.0)
        rows.append({
            'class':             CLASS_NAMES[c],
            'support':           int((y_true == c).sum()),
            'recall':            round(recall,                      4),
            'specificity':       round(specificity,                 4),
            'precision':         round(precision,                   4),
            'f1':                round(f1,                          4),
            'balanced_accuracy': round((recall + specificity) / 2, 4),
            'auroc':             (round(per_class_auroc[c], 4)
                                  if not np.isnan(per_class_auroc[c])
                                  else float('nan')),
        })
    return rows


def plot_confusion_matrix(y_true, y_pred, path: Path, title: str = "") -> None:
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                vmin=0.0, vmax=1.0, annot_kws={"size": 24})
    ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
    ax.set_ylabel('True',      fontsize=14, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved confusion matrix  → {path}")


def plot_roc_curves(
    roc_data,
    per_class_auroc,
    macro_auroc,
    path:              Path,
    optimal_threshold: float | None = None,
    optimal_point:     tuple[float, float] | None = None,
) -> None:
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
              '#ff7f00', '#a65628', '#f781bf', '#999999']
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (fpr, tpr, name) in enumerate(roc_data):
        if fpr is None:
            continue
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{name} (AUC = {per_class_auroc[i]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')

    # Mark the optimal operating point (binary only).
    if optimal_threshold is not None and optimal_point is not None:
        fpr_opt, tpr_opt = optimal_point
        ax.scatter(
            [fpr_opt], [tpr_opt],
            s=120, color='black', marker='*', zorder=5,
            label=f'Optimal (thr={optimal_threshold:.3f})',
        )

    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate',  fontsize=13)
    ax.set_title(f'ROC Curves (Macro AUC = {macro_auroc:.3f})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved ROC curves        → {path}")


# ============================================================================
#  Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AnyMC3D inference on raw NIfTI with full offline preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Required paths ───────────────────────────────────────────────────────
    parser.add_argument('--run_dir',   required=True,
                        help='Run directory containing config.yaml + *.ckpt')
    parser.add_argument('--nifti_dir', required=True,
                        help='Directory of raw .nii / .nii.gz files')

    # ── Optional paths ───────────────────────────────────────────────────────
    parser.add_argument('--labels_csv', default=None,
                        help='CSV with columns [case_id, label] for metrics')
    parser.add_argument('--checkpoint', default=None,
                        help='Specific .ckpt filename (default: best val_auroc)')

    # ── Preprocessing overrides ──────────────────────────────────────────────
    parser.add_argument('--norm', choices=['zscore', 'percentile', 'none'],
                        default=None,
                        help='Override normalization')
    parser.add_argument('--lower_pct', type=float, default=None,
                        help='Override percentile lower bound')
    parser.add_argument('--upper_pct', type=float, default=None,
                        help='Override percentile upper bound')
    parser.add_argument('--target_spacing', type=str, default=None,
                        help='Override target spacing as "H,W,S" (mm)')
    parser.add_argument('--crop_margin', type=int, default=None,
                        help='Override crop margin in voxels')
    parser.add_argument('--target_size', type=int, default=None,
                        help='Override final resize (cube side length)')
    parser.add_argument('--no_crop', action='store_true',
                        help='Disable nonzero-bbox cropping')
    parser.add_argument('--no_resample', action='store_true',
                        help='Disable resampling to target spacing')

    # ── Labels for display ───────────────────────────────────────────────────
    parser.add_argument('--class_names', nargs='+', default=None,
                        help='Override class names (normally from config.yaml)')

    # ── DataLoader ───────────────────────────────────────────────────────────
    parser.add_argument('--batch_size',  type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device',      default='cuda')
    args = parser.parse_args()

    run_dir   = Path(args.run_dir)
    nifti_dir = Path(args.nifti_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")
    if not nifti_dir.exists():
        raise FileNotFoundError(f"nifti_dir not found: {nifti_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    log.info(f"Run directory : {run_dir}")
    log.info(f"NIfTI dir     : {nifti_dir}")
    log.info(f"Device        : {device}")

    # ── Load config ──────────────────────────────────────────────────────────
    cfg = OmegaConf.load(find_config(run_dir))
    log.info(f"Model target  : {cfg.model._target_}")

    # ── Resolve checkpoint ───────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = run_dir / args.checkpoint
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        log.info(f"Checkpoint    : {ckpt_path.name}  (manually specified)")
    else:
        ckpt_path = find_best_checkpoint(run_dir)

    # ── Resolve preprocessing config ─────────────────────────────────────────
    manifest = _load_manifest(run_dir)
    pc       = build_preproc_config(cfg, manifest, args)
    pc.log_summary()

    # ── Resolve num_classes + class names ────────────────────────────────────
    num_classes = resolve_num_classes(cfg)
    global CLASS_NAMES
    CLASS_NAMES = resolve_class_names(cfg, args.class_names, num_classes)
    log.info(f"num_classes   : {num_classes}")
    log.info(f"class_names   : {CLASS_NAMES}")

    # ── Labels (optional) ────────────────────────────────────────────────────
    label_map: dict[str, int] = {}
    has_labels = False
    if args.labels_csv:
        labels_df = pd.read_csv(args.labels_csv)
        if not {'case_id', 'label'}.issubset(labels_df.columns):
            raise ValueError(
                f"labels_csv must have columns [case_id, label]; "
                f"found: {list(labels_df.columns)}"
            )
        labels_df['case_id'] = labels_df['case_id'].astype(str).apply(_strip_channel_suffix)
        label_map  = dict(zip(labels_df['case_id'], labels_df['label'].astype(int)))
        has_labels = True
        log.info(f"Labels loaded : {len(label_map)} entries from {args.labels_csv}")
    else:
        log.info("No labels_csv — predictions only")

    # ── NIfTI discovery ──────────────────────────────────────────────────────
    cases = find_nifti_files(nifti_dir)
    log.info(f"Sample file case_ids  : {[c for c, _ in cases[:3]]}")
    if label_map:
        log.info(f"Sample label_map keys : {list(label_map.keys())[:3]}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset = NiftiInferenceDataset(cases=cases, pc=pc, label_map=label_map)

    # ── Model ────────────────────────────────────────────────────────────────
    log.info("Loading model ...")
    model = load_model(ckpt_path, cfg)

    # ── Inference ────────────────────────────────────────────────────────────
    # If target_size is set, all volumes share a shape and we can batch.
    # Otherwise (crop + resample with no final resize), shapes vary per case
    # and we fall back to batch_size=1.
    log.info(f"Running inference on {len(dataset)} case(s) ...")

    if pc.target_size is not None:
        dataloader = DataLoader(
            dataset,
            batch_size  = args.batch_size,
            shuffle     = False,
            num_workers = args.num_workers,
            pin_memory  = True,
        )
        case_ids, y_raw, probs = run_inference(model, dataloader, device)
    else:
        log.info("target_size not set — using per-case inference (batch_size=1) "
                 "because cropped/resampled shapes may vary across cases.")
        case_ids, y_raw, probs = run_inference_variable_shape(
            model, dataset, device, num_workers=args.num_workers,
        )

    y_pred     = probs.argmax(axis=1)
    confidence = probs.max(axis=1)

    # ── predictions_nifti.csv ────────────────────────────────────────────────
    pred_rows = []
    for i, case_id in enumerate(case_ids):
        row: dict = {
            'case_id':    case_id,
            'pred_label': int(y_pred[i]),
            'pred_class': CLASS_NAMES[int(y_pred[i])],
        }
        if has_labels and y_raw[i] >= 0:
            row['true_label'] = int(y_raw[i])
            row['true_class'] = CLASS_NAMES[int(y_raw[i])]
            row['correct']    = bool(y_raw[i] == y_pred[i])
        for c in range(num_classes):
            row[f'prob_{CLASS_NAMES[c]}'] = round(float(probs[i, c]), 4)
        row['confidence'] = round(float(confidence[i]), 4)
        pred_rows.append(row)

    pred_df  = (pd.DataFrame(pred_rows)
                  .sort_values('case_id')
                  .reset_index(drop=True))
    pred_csv = run_dir / "predictions_nifti.csv"
    pred_df.to_csv(pred_csv, index=False)
    log.info(f"Saved predictions       → {pred_csv}")

    # ── Metrics + plots (only when labels available) ─────────────────────────
    if has_labels:
        mask        = y_raw >= 0
        y_true_eval = y_raw[mask]
        y_pred_eval = y_pred[mask]
        probs_eval  = probs[mask]

        if len(y_true_eval) == 0:
            log.warning(
                "labels_csv provided but no case_ids matched — "
                "check CSV case_id column against NIfTI filenames."
            )
            return

        is_binary = (num_classes == 2)

        # ── Threshold set 1: default argmax (equivalent to P[class=1] >= 0.5) ──
        y_pred_05 = y_pred_eval

        # ── Threshold set 2: Youden's J on ROC (binary only) ──────────────────
        y_pred_opt:        np.ndarray | None = None
        optimal_threshold: float | None     = None
        if is_binary:
            optimal_threshold, youden_j = find_optimal_threshold(
                y_true_eval, probs_eval[:, 1]
            )
            y_pred_opt = (probs_eval[:, 1] >= optimal_threshold).astype(np.int64)
            log.info(f"Optimal threshold (Youden's J): {optimal_threshold:.4f}  "
                     f"(J = {youden_j:.4f})")
        else:
            log.info("Optimal threshold analysis skipped — "
                     "only supported for binary classification.")

        # ── AUROC is threshold-independent; compute once ──────────────────────
        per_class_auroc, macro_auroc, roc_data = compute_auroc(
            y_true_eval, probs_eval, num_classes
        )

        # ── Helper to compute a full metrics bundle for a given prediction set ─
        def _metrics_bundle(y_pred_set: np.ndarray) -> dict:
            return dict(
                accuracy     = accuracy_score(y_true_eval, y_pred_set),
                balanced_acc = compute_balanced_accuracy(
                    y_true_eval, y_pred_set, num_classes
                ),
                f1_macro     = f1_score(
                    y_true_eval, y_pred_set, average='macro', zero_division=0
                ),
                class_stats  = compute_per_class_stats(
                    y_true_eval, y_pred_set, per_class_auroc, num_classes
                ),
            )

        m_05 = _metrics_bundle(y_pred_05)
        m_opt = _metrics_bundle(y_pred_opt) if y_pred_opt is not None else None

        # ── Console summary ───────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"  RESULTS  ({len(y_true_eval)} cases with labels)")
        print(f"{'='*60}")
        print(f"  Macro AUROC:       {macro_auroc:.4f}   (threshold-independent)")
        print(f"\n  ── Default threshold (argmax / 0.5) ─────────────────────")
        print(f"  Accuracy:          {m_05['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {m_05['balanced_acc']:.4f}")
        print(f"  F1 (macro):        {m_05['f1_macro']:.4f}")
        if m_opt is not None and optimal_threshold is not None:
            print(f"\n  ── Optimal threshold (Youden's J = {optimal_threshold:.4f}) ──")
            print(f"  Accuracy:          {m_opt['accuracy']:.4f}  "
                  f"(Δ {m_opt['accuracy'] - m_05['accuracy']:+.4f})")
            print(f"  Balanced Accuracy: {m_opt['balanced_acc']:.4f}  "
                  f"(Δ {m_opt['balanced_acc'] - m_05['balanced_acc']:+.4f})")
            print(f"  F1 (macro):        {m_opt['f1_macro']:.4f}  "
                  f"(Δ {m_opt['f1_macro'] - m_05['f1_macro']:+.4f})")
        print(f"\n  Per-class AUROC:")
        for c, name in enumerate(CLASS_NAMES):
            v = per_class_auroc[c]
            print(f"    {name}: {v:.4f}" if not np.isnan(v) else f"    {name}: N/A")
        print(f"{'='*60}\n")

        # ── summary_nifti.csv ─────────────────────────────────────────────────
        summary_rows = [
            {'metric': 'checkpoint',        'value': ckpt_path.name},
            {'metric': 'model_target',      'value': cfg.model._target_},
            {'metric': 'norm',              'value': pc.norm},
            {'metric': 'lower_pct',         'value': pc.lower_pct if pc.norm == 'percentile' else None},
            {'metric': 'upper_pct',         'value': pc.upper_pct if pc.norm == 'percentile' else None},
            {'metric': 'target_spacing_mm', 'value': list(pc.target_spacing) if pc.target_spacing else None},
            {'metric': 'crop_margin_vox',   'value': pc.crop_margin if pc.do_crop else None},
            {'metric': 'target_size',       'value': pc.target_size},
            {'metric': 'n_classes',         'value': num_classes},
            {'metric': 'class_names',       'value': str(CLASS_NAMES)},
            {'metric': 'nifti_dir',         'value': str(nifti_dir)},
            {'metric': 'n_cases',           'value': len(y_true_eval)},
            # ── Threshold-independent ─────────────────────────────────────────
            {'metric': 'macro_auroc',       'value': round(macro_auroc, 4)},
            # ── thr=0.5 metrics ───────────────────────────────────────────────
            {'metric': 'thr_default',                'value': 0.5},
            {'metric': 'accuracy_thr0.5',            'value': round(m_05['accuracy'],     4)},
            {'metric': 'balanced_accuracy_thr0.5',   'value': round(m_05['balanced_acc'], 4)},
            {'metric': 'f1_macro_thr0.5',            'value': round(m_05['f1_macro'],     4)},
        ]
        for row in m_05['class_stats']:
            name = row['class']
            summary_rows += [
                {'metric': f'{name}_support',                  'value': row['support']},
                {'metric': f'{name}_recall_thr0.5',            'value': row['recall']},
                {'metric': f'{name}_specificity_thr0.5',       'value': row['specificity']},
                {'metric': f'{name}_precision_thr0.5',         'value': row['precision']},
                {'metric': f'{name}_f1_thr0.5',                'value': row['f1']},
                {'metric': f'{name}_balanced_accuracy_thr0.5', 'value': row['balanced_accuracy']},
                {'metric': f'{name}_auroc',                    'value': row['auroc']},
            ]

        # ── Optimal-threshold metrics (binary only) ───────────────────────────
        if m_opt is not None and optimal_threshold is not None:
            summary_rows += [
                {'metric': 'thr_optimal',                'value': round(optimal_threshold, 4)},
                {'metric': 'accuracy_thrOpt',            'value': round(m_opt['accuracy'],     4)},
                {'metric': 'balanced_accuracy_thrOpt',   'value': round(m_opt['balanced_acc'], 4)},
                {'metric': 'f1_macro_thrOpt',            'value': round(m_opt['f1_macro'],     4)},
            ]
            for row in m_opt['class_stats']:
                name = row['class']
                summary_rows += [
                    {'metric': f'{name}_recall_thrOpt',            'value': row['recall']},
                    {'metric': f'{name}_specificity_thrOpt',       'value': row['specificity']},
                    {'metric': f'{name}_precision_thrOpt',         'value': row['precision']},
                    {'metric': f'{name}_f1_thrOpt',                'value': row['f1']},
                    {'metric': f'{name}_balanced_accuracy_thrOpt', 'value': row['balanced_accuracy']},
                ]

        summary_df  = pd.DataFrame(summary_rows)
        summary_csv = run_dir / "summary_nifti.csv"
        summary_df.to_csv(summary_csv, index=False)
        log.info(f"Saved summary           → {summary_csv}")

        # ── predictions CSV: add the optimal-threshold prediction column ──────
        if y_pred_opt is not None:
            # Build a lookup from case_id → thrOpt prediction for the subset
            # with labels (the rest remain unlabeled and keep thr=0.5 only).
            case_ids_arr = np.array(case_ids)
            labeled_ids  = case_ids_arr[mask]
            opt_lookup   = dict(zip(labeled_ids, y_pred_opt))

            pred_df['pred_label_thrOpt'] = pred_df['case_id'].map(
                lambda c: int(opt_lookup[c]) if c in opt_lookup else -1
            )
            pred_df['pred_class_thrOpt'] = pred_df['pred_label_thrOpt'].map(
                lambda lbl: CLASS_NAMES[lbl] if lbl >= 0 else ''
            )
            pred_df.to_csv(pred_csv, index=False)
            log.info(f"Updated predictions with thrOpt column → {pred_csv}")

        # ── Plots ─────────────────────────────────────────────────────────────
        plot_confusion_matrix(
            y_true_eval, y_pred_05,
            run_dir / "confusion_matrix_thr0.5.png",
            title=f"Confusion Matrix (threshold = 0.5)",
        )
        if y_pred_opt is not None and optimal_threshold is not None:
            plot_confusion_matrix(
                y_true_eval, y_pred_opt,
                run_dir / "confusion_matrix_thrOpt.png",
                title=f"Confusion Matrix (optimal threshold = {optimal_threshold:.3f})",
            )

        # Compute the (FPR, TPR) of the optimal threshold for annotation on ROC.
        optimal_point: tuple[float, float] | None = None
        if y_pred_opt is not None:
            n_pos = int((y_true_eval == 1).sum())
            n_neg = int((y_true_eval == 0).sum())
            if n_pos > 0 and n_neg > 0:
                tp_opt = int(((y_pred_opt == 1) & (y_true_eval == 1)).sum())
                fp_opt = int(((y_pred_opt == 1) & (y_true_eval == 0)).sum())
                optimal_point = (fp_opt / n_neg, tp_opt / n_pos)

        plot_roc_curves(
            roc_data, per_class_auroc, macro_auroc,
            run_dir / "roc_curves_nifti.png",
            optimal_threshold=optimal_threshold if is_binary else None,
            optimal_point=optimal_point,
        )

    log.info(f"Done. Outputs in: {run_dir}")


if __name__ == "__main__":
    main()