"""
Inference Script for AnyMC3D / RSNA-CNN

Point to a run directory and it auto-discovers the config and best checkpoint:

    python inference.py --run_dir /path/to/checkpoints/anymc3d-vitb14-t1c_CosineLR

Outputs saved inside the run directory:
  - predictions_{split}.csv       per-case predictions + probabilities + confidence
  - summary_{split}.csv           overall + per-class metrics
  - confusion_matrix_{split}.png
  - roc_curves_{split}.png

Optional overrides:
    python inference.py --run_dir ... --split val
    python inference.py --run_dir ... --checkpoint specific_epoch.ckpt  # pick a specific ckpt
    python inference.py --run_dir ... --fold 1 --data_root /new/path
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    f1_score, accuracy_score
)
from sklearn.preprocessing import label_binarize
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

CLASS_NAMES = ['COCA1', 'COCA2', 'COCA3', 'COCA4']


# ---------------------------------------------------------------
# Auto-discovery helpers
# ---------------------------------------------------------------

def find_config(run_dir: Path) -> Path:
    """Look for config.yaml in the run directory."""
    candidate = run_dir / "config.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"No config.yaml found in {run_dir}.\n"
        f"Expected: {candidate}"
    )


def find_best_checkpoint(run_dir: Path) -> Path:
    """
    Find the best checkpoint in the run directory.
    Picks the .ckpt with the highest val_auroc value encoded in its filename.
    Falls back to the most recently modified .ckpt if no auroc found in name.
    """
    ckpts = sorted(run_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files found in {run_dir}")

    # Try to parse val_auroc from filename: epoch=XX-val_auroc=0.XXXX.ckpt
    scored = []
    for ckpt in ckpts:
        try:
            auroc_str = ckpt.stem.split("val_auroc=")[1]
            scored.append((float(auroc_str), ckpt))
        except (IndexError, ValueError):
            pass

    if scored:
        best = max(scored, key=lambda x: x[0])[1]
        print(f"Auto-selected best checkpoint (val_auroc={max(scored, key=lambda x: x[0])[0]:.4f}):")
    else:
        # fallback: most recently modified
        best = max(ckpts, key=lambda p: p.stat().st_mtime)
        print(f"Auto-selected most recent checkpoint (no auroc in filename):")

    print(f"  {best.name}")
    return best


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------

def compute_balanced_accuracy(y_true, y_pred, num_classes):
    bal_accs = []
    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        tn = ((y_pred != c) & (y_true != c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        bal_accs.append((recall + specificity) / 2)
    return float(np.mean(bal_accs))


def compute_auroc(y_true, probs, num_classes):
    # label_binarize returns shape (N, 1) for binary — expand to (N, 2)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))
    if num_classes == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])

    per_class = []
    roc_data  = []
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


def compute_per_class_stats(y_true, y_pred, per_class_auroc, num_classes):
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
        f1          = 2 * precision * recall / (precision + recall) \
                      if (precision + recall) > 0 else 0.0
        auroc_v     = per_class_auroc[c]
        rows.append({
            'class':             CLASS_NAMES[c],
            'support':           int((y_true == c).sum()),
            'recall':            round(recall,      4),
            'specificity':       round(specificity, 4),
            'precision':         round(precision,   4),
            'f1':                round(f1,          4),
            'balanced_accuracy': round((recall + specificity) / 2, 4),
            'auroc':             round(auroc_v, 4) if not np.isnan(auroc_v) else float('nan'),
        })
    return rows


# ---------------------------------------------------------------
# Plots
# ---------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, path):
    cm      = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)  # row-normalised (% of true class)

    # Build annotation: count on top line, percentage below
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)"

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", ax=ax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                vmin=0.0, vmax=1.0,
                annot_kws={"size": 16})
    ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
    ax.set_ylabel('True',      fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved confusion matrix  -> {path}")


def plot_roc_curves(roc_data, per_class_auroc, macro_auroc, path):
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (fpr, tpr, name) in enumerate(roc_data):
        if fpr is None:
            continue
        ax.plot(fpr, tpr, color=colors[i], lw=2,
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
    print(f"Saved ROC curves        -> {path}")


# ---------------------------------------------------------------
# Model loading & inference
# ---------------------------------------------------------------

def load_model(checkpoint_path, cfg):
    arch = cfg.model.arch
    if arch == "anymc3d":
        from anymc3d import AnyMC3DLightningModule
        model = AnyMC3DLightningModule.load_from_checkpoint(
            str(checkpoint_path), map_location='cpu')
    elif arch == "rsna_cnn":
        from rsna_kaggle_model import RSNAKaggleLightningModule
        model = RSNAKaggleLightningModule.load_from_checkpoint(
            str(checkpoint_path), map_location='cpu')
    else:
        raise ValueError(f"Unknown arch: {arch}")
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, dataloader, device, arch):
    model.to(device)
    model.eval()
    all_case_ids, all_labels, all_probs = [], [], []
    for volumes, labels, case_ids in dataloader:
        volumes = volumes.to(device)
        if arch == "anymc3d":
            logits, _ = model.model({model.modalities[0]: volumes})
        else:
            logits = model.model(volumes)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_case_ids.extend(list(case_ids))
        all_labels.extend(labels.numpy().tolist())
        all_probs.append(probs)
    return all_case_ids, np.array(all_labels), np.concatenate(all_probs, axis=0)


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir',     required=True,
                        help='Path to the run directory, e.g. checkpoints/anymc3d-vitb14-t1c')
    parser.add_argument('--checkpoint',  default=None,
                        help='Specific .ckpt filename inside run_dir (default: auto-select best)')
    parser.add_argument('--split',       default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--fold',        type=int, default=None)
    parser.add_argument('--data_root',   default=None)
    parser.add_argument('--batch_size',  type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device',      default='cuda')
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"\nRun directory : {run_dir}")
    print(f"Device        : {device}")
    print(f"Split         : {args.split}")

    # ── Auto-discover config and checkpoint ───────────────────────────────────
    config_path = find_config(run_dir)
    print(f"Config        : {config_path.name}")

    if args.checkpoint:
        ckpt_path = run_dir / args.checkpoint
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        print(f"Checkpoint    : {ckpt_path.name}  (manually specified)")
    else:
        ckpt_path = find_best_checkpoint(run_dir)

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = OmegaConf.load(config_path)
    if args.data_root:
        cfg.data.data_root = args.data_root
    if args.fold is not None:
        cfg.data.fold = args.fold
    print(f"Arch          : {cfg.model.arch}  |  Fold: {cfg.data.fold}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model...")
    model = load_model(ckpt_path, cfg)

    # ── Data — branch on dataset type ─────────────────────────────────────────
    dataset_type = cfg.data.get('dataset', 'meningioma')

    if dataset_type == 'pdcad':
        from pdcad_dataset import PDCADDataset
        global CLASS_NAMES
        CLASS_NAMES = ['Class0', 'Class1']
        dataset = PDCADDataset(
            data_root   = cfg.data.data_root,
            labels_path = cfg.data.labels_path,
            splits_path = cfg.data.splits_path,
            split       = args.split,
            fold        = cfg.data.fold,
            augment     = False,
        )
    else:
        from meningioma_holdout_dataset import MeningiomaDataset
        dataset = MeningiomaDataset(
            data_root   = cfg.data.data_root,
            labels_path = cfg.data.labels_path,
            splits_path = cfg.data.splits_path,
            split       = args.split,
            fold        = cfg.data.fold,
            augment     = False,
        )

    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"Running inference on {len(dataset)} cases...")
    case_ids, y_true, probs = run_inference(model, dataloader, device, cfg.model.arch)

    y_pred      = probs.argmax(axis=1)
    num_classes = probs.shape[1]
    confidence  = probs.max(axis=1)

    # ── Metrics ───────────────────────────────────────────────────────────────
    accuracy     = accuracy_score(y_true, y_pred)
    balanced_acc = compute_balanced_accuracy(y_true, y_pred, num_classes)
    f1_macro     = f1_score(y_true, y_pred, average='macro', zero_division=0)
    per_class_auroc, macro_auroc, roc_data = compute_auroc(y_true, probs, num_classes)
    class_stats = compute_per_class_stats(y_true, y_pred, per_class_auroc, num_classes)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  RESULTS — {args.split.upper()} SET")
    print(f"{'='*50}")
    print(f"  Cases:             {len(y_true)}")
    print(f"  Accuracy:          {accuracy:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  Macro AUROC:       {macro_auroc:.4f}")
    print(f"  F1 (macro):        {f1_macro:.4f}")
    print(f"\n  Per-class AUROC:")
    for c, name in enumerate(CLASS_NAMES):
        v = per_class_auroc[c]
        print(f"    {name}: {v:.4f}" if not np.isnan(v) else f"    {name}: N/A")
    print(f"{'='*50}\n")

    # ── predictions.csv ───────────────────────────────────────────────────────
    pred_rows = []
    for i, case_id in enumerate(case_ids):
        row = {
            'case_id':    case_id,
            'true_label': int(y_true[i]),
            'true_class': CLASS_NAMES[int(y_true[i])],
            'pred_label': int(y_pred[i]),
            'pred_class': CLASS_NAMES[int(y_pred[i])],
        }
        for c in range(num_classes):
            row[f'prob_{CLASS_NAMES[c]}'] = round(float(probs[i, c]), 4)
        row['confidence'] = round(float(confidence[i]), 4)
        row['correct']    = bool(y_true[i] == y_pred[i])
        pred_rows.append(row)

    predictions_df = (pd.DataFrame(pred_rows)
                        .sort_values('case_id')
                        .reset_index(drop=True))
    pred_csv = run_dir / f"predictions_{args.split}.csv"
    predictions_df.to_csv(pred_csv, index=False)
    print(f"Saved predictions       -> {pred_csv}")

    # ── summary.csv ───────────────────────────────────────────────────────────
    summary_rows = [
        {'metric': 'split',             'value': args.split},
        {'metric': 'checkpoint',        'value': ckpt_path.name},
        {'metric': 'n_cases',           'value': len(y_true)},
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

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = run_dir / f"summary_{args.split}.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary           -> {summary_csv}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(
        y_true, y_pred,
        run_dir / f"confusion_matrix_{args.split}.png"
    )
    plot_roc_curves(
        roc_data, per_class_auroc, macro_auroc,
        run_dir / f"roc_curves_{args.split}.png"
    )

    print(f"\nDone. All outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()