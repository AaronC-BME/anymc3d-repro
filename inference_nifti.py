"""
inference_nifti.py — AnyMC3D / V-JEPA 2.1 inference on raw NIfTI files
=======================================================================

Point to a run directory (config + checkpoint) and a folder of NIfTI images.
Preprocessing is applied inline, exactly matching the offline training pipeline.

────────────────────────────────────────────────────────────────────────────────
USAGE — prediction only (no ground-truth)
────────────────────────────────────────────────────────────────────────────────
  python inference_nifti.py \\
      --run_dir   checkpoints/anymc3d-vitb14-pdcad \\
      --nifti_dir /data/PDCAD/raw_nifti

────────────────────────────────────────────────────────────────────────────────
USAGE — with ground-truth labels (metrics + confusion matrix + ROC curves)
────────────────────────────────────────────────────────────────────────────────
  python inference_nifti.py \\
      --run_dir   checkpoints/anymc3d-vitb14-pdcad \\
      --nifti_dir /data/PDCAD/raw_nifti \\
      --labels_csv /data/PDCAD/labels.csv          # columns: case_id, label

────────────────────────────────────────────────────────────────────────────────
USAGE — V-JEPA 2.1 checkpoint (on a machine where the original .pt path differs)
────────────────────────────────────────────────────────────────────────────────
  python inference_nifti.py \\
      --run_dir   checkpoints/vjepa21-vitb-pdcad \\
      --nifti_dir /data/PDCAD/raw_nifti \\
      --labels_csv /data/PDCAD/labels.csv

  The --vjepa_base_weights flag is NOT needed for inference: the Lightning
  checkpoint already contains the full model state (base encoder + LoRA +
  head). It is only needed if you want to re-run __init__ with base weights
  for debugging purposes.

────────────────────────────────────────────────────────────────────────────────
NIfTI LAYOUT (either flat or one-level subdirectories)
────────────────────────────────────────────────────────────────────────────────
  Flat:
      nifti_dir/
        RJPD_000_0000.nii.gz
        RJPD_001_0000.nii.gz
        ...
  Subdirectory (subject ID = parent folder name):
      nifti_dir/
        RJPD_000/RJPD_000_0000.nii.gz
        RJPD_001/RJPD_001_0000.nii.gz
        ...

────────────────────────────────────────────────────────────────────────────────
PREPROCESSING (applied inline, matching the offline pipeline)
────────────────────────────────────────────────────────────────────────────────
  pdcad      : MONAI ScaleIntensityRangePercentiles (p0 → p99.5) → [0, 1]
               No spatial resampling — native 300×300×70 passed to model.
  meningioma : Z-score normalisation within brain mask → trilinear resample to 256³

────────────────────────────────────────────────────────────────────────────────
OUTPUTS  (saved inside run_dir)
────────────────────────────────────────────────────────────────────────────────
  predictions_nifti.csv          — per-case predictions + class probs
  summary_nifti.csv              — overall + per-class metrics   (labels only)
  confusion_matrix_nifti.png     — confusion matrix               (labels only)
  roc_curves_nifti.png           — per-class ROC curves           (labels only)


Last run:
    python inference_nifti.py \
      --run_dir   checkpoints/anymc3d_2VisBlck_LoRALR_1e-4_Headlr_1e-3_vitb_pdcad_PS_308_308_70_150ep_fold0\
      --nifti_dir /home/jma/Documents/projects/safwat/Datasets/Dataset031_PDCAD_NM/imagesVal \
      --labels_csv /home/jma/Documents/projects/safwat/Datasets/Dataset031_PDCAD_NM/val_cases.csv \
      --checkpoint "epoch=87-val_auroc=0.9132.ckpt" 
"""

from __future__ import annotations

import argparse
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

# ── Default class names (overridden by dataset type at runtime) ──────────────
CLASS_NAMES: list[str] = ['COCA1', 'COCA2', 'COCA3', 'COCA4']

# ── Target spatial size for meningioma resampling ────────────────────────────
MENINGIOMA_TARGET = (256, 256, 256)


# ============================================================================
#  Helpers
# ============================================================================

def _strip_channel_suffix(s: str) -> str:
    """
    Normalise a case ID by stripping the nnU-Net channel suffix (_0000).
    Works whether the suffix is present or not, so both CSV rows and
    NIfTI filenames map to the same key.
      'RJPD_000_0000' → 'RJPD_000'
      'RJPD_000'      → 'RJPD_000'
    """
    return s[:-5] if s.endswith("_0000") else s


# ============================================================================
#  Auto-discovery helpers
# ============================================================================

def find_config(run_dir: Path) -> Path:
    c = run_dir / "config.yaml"
    if c.exists():
        return c
    raise FileNotFoundError(f"config.yaml not found in {run_dir}")


def find_best_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("*.ckpt"))
    if not ckpts:
        # Also check one level deep (Lightning saves to epoch=N-step=M.ckpt subdirs)
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
    """
    Discover NIfTI files (.nii or .nii.gz) in a flat folder or one level of
    subdirectories. Returns [(case_id, path), ...] sorted by case_id.

    Case IDs are normalised via _strip_channel_suffix so that both
    'RJPD_000_0000.nii.gz' and 'RJPD_000.nii.gz' yield 'RJPD_000',
    matching labels CSVs regardless of whether they include the suffix.
    """
    hits: list[tuple[str, Path]] = []

    # Flat layout
    for ext in ("*.nii.gz", "*.nii"):
        for p in sorted(nifti_dir.glob(ext)):
            stem    = p.name.replace(".nii.gz", "").replace(".nii", "")
            case_id = _strip_channel_suffix(stem)
            hits.append((case_id, p))

    # One-level subdirectory layout
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
            f"(checked flat layout and one-level subdirectories)"
        )

    # Deduplicate by case_id, keeping first occurrence
    seen: set[str] = set()
    unique: list[tuple[str, Path]] = []
    for case_id, p in hits:
        if case_id not in seen:
            seen.add(case_id)
            unique.append((case_id, p))

    log.info(f"Found {len(unique)} NIfTI file(s) in {nifti_dir}")
    return unique


# ============================================================================
#  Preprocessing
# ============================================================================

def load_nifti(path: Path) -> np.ndarray:
    """Load a NIfTI file and return a float32 numpy array with shape (H, W, D)."""
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required: pip install nibabel")
    img = nib.load(str(path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    # Ensure (H, W, D) — drop any extra singleton dims
    while vol.ndim > 3:
        vol = vol.squeeze(-1)
    return vol


def preprocess_pdcad(vol: np.ndarray) -> np.ndarray:
    """
    PDCAD (Nuclear Medicine) preprocessing — matches offline training pipeline:
      ScaleIntensityRangePercentiles(lower=0, upper=99.5, b_min=0, b_max=1, clip=True)

    No spatial resampling: the model's internal _prepare_clip handles in-plane
    resize (300 → crop_size: 384 for V-JEPA 2.1, 308 for DINOv2 ViT-B/14).
    """
    p_low  = np.percentile(vol, 0)
    p_high = np.percentile(vol, 99.5)
    if p_high - p_low < 1e-8:
        return np.zeros_like(vol)
    vol = (vol - p_low) / (p_high - p_low)
    return np.clip(vol, 0.0, 1.0).astype(np.float32)


def preprocess_meningioma(vol: np.ndarray) -> np.ndarray:
    """
    Meningioma (T1c MRI) preprocessing — matches offline training pipeline:
      1. Brain-mask foreground (voxels > 0)
      2. Z-score normalise within mask; set background to 0
      3. Trilinear resample to 256³

    The output is a z-scored volume; background = 0.
    The model's forward pass does not assume [0, 1] values for meningioma.
    """
    # ── Z-score on foreground ─────────────────────────────────────────────────
    mask = vol > 0
    if mask.sum() > 100:
        mu  = float(vol[mask].mean())
        sig = float(vol[mask].std())
        if sig > 1e-6:
            vol = (vol - mu) / sig
        vol[~mask] = 0.0
    else:
        vol = np.zeros_like(vol)

    # ── Trilinear resample to 256³ ────────────────────────────────────────────
    th, tw, td = MENINGIOMA_TARGET
    if vol.shape != (th, tw, td):
        t = torch.from_numpy(vol[None, None])          # (1, 1, H, W, D)
        t = F.interpolate(
            t,
            size=(th, tw, td),
            mode='trilinear',
            align_corners=False,
        )
        vol = t.squeeze().numpy()

    return vol.astype(np.float32)


# ============================================================================
#  Dataset
# ============================================================================

class NiftiInferenceDataset(Dataset):
    """
    Minimal dataset that loads NIfTI files on the fly and applies the
    appropriate preprocessing for the target model.

    Returns:
        volume  : torch.Tensor  shape (1, H, W, S) — single-channel, no batch dim
        label   : int   (-1 if unknown)
        case_id : str
    """

    def __init__(
        self,
        cases:        list[tuple[str, Path]],
        dataset_type: str,
        label_map:    dict[str, int] | None = None,
    ):
        self.cases        = cases
        self.dataset_type = dataset_type
        self.label_map    = label_map or {}

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int):
        case_id, path = self.cases[idx]
        label = self.label_map.get(case_id, -1)

        vol = load_nifti(path)

        if self.dataset_type == 'pdcad':
            vol = preprocess_pdcad(vol)
        else:
            vol = preprocess_meningioma(vol)

        # Add channel dim → (1, H, W, D)
        tensor = torch.from_numpy(vol[None])
        return tensor, label, case_id


# ============================================================================
#  Model loading
# ============================================================================

def _resolve_class_from_target(target: str):
    """
    Resolve a dotted _target_ string to a Python class.
    e.g. 'model_arch.anymc3d.AnyMC3DLightningModule'
         → importlib.import_module('model_arch.anymc3d').AnyMC3DLightningModule
    """
    import importlib
    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def load_model(ckpt_path: Path, cfg):
    """
    Load a Lightning checkpoint using cfg.model._target_ to resolve the class.

    This works directly with the config.yaml that Hydra saves during training —
    no separate 'arch' key is needed.

    For V-JEPA models: vjepa_checkpoint_path is overridden to None so __init__
    skips pretrained base-weight loading.  Lightning restores the full model
    state from the .ckpt file regardless.
    """
    target = cfg.model._target_
    log.info(f"Resolving model class from _target_: {target}")
    LightningClass = _resolve_class_from_target(target)

    is_vjepa = "vjepa2" in target.lower()

    if is_vjepa:
        model = LightningClass.load_from_checkpoint(
            str(ckpt_path),
            map_location="cpu",
            vjepa_checkpoint_path=None,   # skip redundant base-weight loading
        )
    else:
        model = LightningClass.load_from_checkpoint(
            str(ckpt_path),
            map_location="cpu",
        )

    model.eval()
    return model


# ============================================================================
#  Inference
# ============================================================================

@torch.no_grad()
def run_inference(
    model,
    dataloader: DataLoader,
    device:     torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()

    # Duck-type: anymc3d variants expose model.modalities and expect a dict;
    # vjepa2_anymc3d and others take a plain tensor.
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


# ============================================================================
#  Metrics
# ============================================================================

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

    valid  = [v for v in per_class if not np.isnan(v)]
    macro  = float(np.mean(valid)) if valid else float('nan')
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
            'recall':            round(recall,                          4),
            'specificity':       round(specificity,                     4),
            'precision':         round(precision,                       4),
            'f1':                round(f1,                              4),
            'balanced_accuracy': round((recall + specificity) / 2,     4),
            'auroc':             (round(per_class_auroc[c], 4)
                                  if not np.isnan(per_class_auroc[c])
                                  else float('nan')),
        })
    return rows


# ============================================================================
#  Plots
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, path: Path) -> None:
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
    ax.set_title('', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved confusion matrix  → {path}")


def plot_roc_curves(roc_data, per_class_auroc, macro_auroc, path: Path) -> None:
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (fpr, tpr, name) in enumerate(roc_data):
        if fpr is None:
            continue
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f'{name} (AUC = {per_class_auroc[i]:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
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
        description="AnyMC3D / V-JEPA 2.1 inference on raw NIfTI images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--run_dir', required=True,
        help='Path to the run directory that contains config.yaml + *.ckpt files'
    )
    parser.add_argument(
        '--nifti_dir', required=True,
        help='Directory containing raw NIfTI files (.nii or .nii.gz)'
    )
    parser.add_argument(
        '--labels_csv', default=None,
        help=(
            'Optional CSV with columns [case_id, label] for metric computation. '
            'Labels are integers (0-based). If omitted, predictions only.'
        )
    )
    parser.add_argument(
        '--checkpoint', default=None,
        help='Specific .ckpt filename inside run_dir (default: auto-select best val_auroc)'
    )
    parser.add_argument(
        '--dataset', default=None, choices=['pdcad', 'meningioma'],
        help=(
            'Override dataset type for preprocessing. '
            'Normally auto-detected from config.yaml (cfg.data.dataset).'
        )
    )
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

    # ── Config + checkpoint ──────────────────────────────────────────────────
    config_path = find_config(run_dir)
    log.info(f"Config        : {config_path.name}")

    if args.checkpoint:
        ckpt_path = run_dir / args.checkpoint
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        log.info(f"Checkpoint    : {ckpt_path.name}  (manually specified)")
    else:
        ckpt_path = find_best_checkpoint(run_dir)

    cfg = OmegaConf.load(config_path)

    # ── Dataset type ─────────────────────────────────────────────────────────
    dataset_type: str = (
        args.dataset
        or cfg.data.get('dataset', 'meningioma')
    )
    log.info(f"Dataset type  : {dataset_type}")
    log.info(f"Model target  : {cfg.model._target_}")

    # ── Class names ──────────────────────────────────────────────────────────
    global CLASS_NAMES
    if dataset_type == 'pdcad':
        CLASS_NAMES = ['Class0', 'Class1']
    else:
        CLASS_NAMES = ['COCA1', 'COCA2', 'COCA3', 'COCA4']

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
        log.info("No labels_csv provided — predictions only (no metrics)")

    # ── NIfTI discovery ──────────────────────────────────────────────────────
    cases = find_nifti_files(nifti_dir)

    log.info(f"Sample file case_ids  : {[c for c, _ in cases[:3]]}")
    if label_map:
        log.info(f"Sample label_map keys : {list(label_map.keys())[:3]}")

    # ── Dataset + DataLoader ─────────────────────────────────────────────────
    dataset = NiftiInferenceDataset(
        cases        = cases,
        dataset_type = dataset_type,
        label_map    = label_map,
    )
    dataloader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
    )

    # ── Load model ───────────────────────────────────────────────────────────
    log.info("Loading model ...")
    model = load_model(ckpt_path, cfg)

    # ── Inference ────────────────────────────────────────────────────────────
    log.info(f"Running inference on {len(dataset)} case(s) ...")
    case_ids, y_raw, probs = run_inference(model, dataloader, device)

    y_pred      = probs.argmax(axis=1)
    num_classes = probs.shape[1]
    confidence  = probs.max(axis=1)

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

    # ── Metrics + plots (only when labels are available) ─────────────────────
    if has_labels:
        mask        = y_raw >= 0
        y_true_eval = y_raw[mask]
        y_pred_eval = y_pred[mask]
        probs_eval  = probs[mask]

        if len(y_true_eval) == 0:
            log.warning(
                "labels_csv provided but none of the case_ids matched — "
                "check that the CSV case_id column matches the NIfTI filenames. "
                "Skipping metrics."
            )
            return

        accuracy     = accuracy_score(y_true_eval, y_pred_eval)
        balanced_acc = compute_balanced_accuracy(y_true_eval, y_pred_eval, num_classes)
        f1_macro     = f1_score(y_true_eval, y_pred_eval, average='macro', zero_division=0)
        per_class_auroc, macro_auroc, roc_data = compute_auroc(
            y_true_eval, probs_eval, num_classes)
        class_stats = compute_per_class_stats(
            y_true_eval, y_pred_eval, per_class_auroc, num_classes)

        print(f"\n{'='*54}")
        print(f"  RESULTS  ({len(y_true_eval)} cases with labels)")
        print(f"{'='*54}")
        print(f"  Accuracy:          {accuracy:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        print(f"  Macro AUROC:       {macro_auroc:.4f}")
        print(f"  F1 (macro):        {f1_macro:.4f}")
        print(f"\n  Per-class AUROC:")
        for c, name in enumerate(CLASS_NAMES):
            v = per_class_auroc[c]
            print(f"    {name}: {v:.4f}" if not np.isnan(v) else f"    {name}: N/A")
        print(f"{'='*54}\n")

        summary_rows = [
            {'metric': 'checkpoint',        'value': ckpt_path.name},
            {'metric': 'dataset_type',      'value': dataset_type},
            {'metric': 'nifti_dir',         'value': str(nifti_dir)},
            {'metric': 'n_cases',           'value': len(y_true_eval)},
            {'metric': 'accuracy',          'value': round(accuracy,     4)},
            {'metric': 'balanced_accuracy', 'value': round(balanced_acc, 4)},
            {'metric': 'macro_auroc',       'value': round(macro_auroc,  4)},
            {'metric': 'f1_macro',          'value': round(f1_macro,     4)},
        ]
        for row in class_stats:
            name = row['class']
            summary_rows += [
                {'metric': f'{name}_support',           'value': row['support']},
                {'metric': f'{name}_recall',            'value': row['recall']},
                {'metric': f'{name}_specificity',       'value': row['specificity']},
                {'metric': f'{name}_precision',         'value': row['precision']},
                {'metric': f'{name}_f1',                'value': row['f1']},
                {'metric': f'{name}_balanced_accuracy', 'value': row['balanced_accuracy']},
                {'metric': f'{name}_auroc',             'value': row['auroc']},
            ]
        summary_df  = pd.DataFrame(summary_rows)
        summary_csv = run_dir / "summary_nifti.csv"
        summary_df.to_csv(summary_csv, index=False)
        log.info(f"Saved summary           → {summary_csv}")

        plot_confusion_matrix(
            y_true_eval, y_pred_eval,
            run_dir / "confusion_matrix_nifti.png",
        )
        plot_roc_curves(
            roc_data, per_class_auroc, macro_auroc,
            run_dir / "roc_curves_nifti.png",
        )

    log.info(f"Done. Outputs in: {run_dir}")


if __name__ == "__main__":
    main()