# AnyMC3D

Scalable 3D Medical Image Classifier adapted from 2D Foundation Models.

Based on: *"Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification"* — Liu et al., 2025 ([arXiv:2512.12887](https://arxiv.org/abs/2512.12887))

This implementation applies AnyMC3D to binary classification on the **PDCAD** dataset using T1-contrast MRI volumes.

---

## How it works

AnyMC3D wraps a frozen DINOv2 backbone with lightweight LoRA adapters (~1.2M trainable parameters). For each 3D volume:

1. All 2D slices are extracted along the axial, coronal, and sagittal planes
2. Each slice is encoded with the LoRA-adapted DINOv2 (ViT-B/14 or ViT-L/14)
3. An attention pooling module aggregates slices within each plane
4. The three plane embeddings are mean-fused into a single volume embedding
5. A linear head produces the final class prediction

---

## Project structure

```
AnyMC3D/
├── anymc3d.py              # Model definition (AnyMC3D + Lightning module)
├── train.py                # Hydra-driven training script
├── inference.py            # Inference + metrics + plots
├── pdcad_dataset.py        # PDCAD Dataset and DataModule
├── balanced_accuracy.py    # Custom balanced accuracy metric
├── requirements.txt        # Python dependencies
└── configs/
    ├── train_pdcad.yaml    # Top-level PDCAD training config
    └── model/
        ├── anymc3d_vitb_pdcad.yaml   # ViT-B model config for PDCAD
        └── anymc3d_vitl.yaml         # ViT-L model config
```

---

## Installation

**1. Install PyTorch (with CUDA) first:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**2. Install remaining dependencies:**

```bash
pip install -r requirements.txt
```

---

## PDCAD dataset format

The dataset loader expects the following directory structure:

```
<data_root>/
    <case_id>/
        ses-DEFAULT/
            <case_id>_0000.b2nd     # Blosc2-compressed volume (1, D, H, W) float32
```

Two JSON metadata files are also required:

**`labels.json`** — maps each case ID to its binary label:
```json
{
    "case_001": 0,
    "case_002": 1,
    "case_003": 0
}
```

**`splits.json`** — defines train/val/test splits per fold:
```json
{
    "0": {
        "train": ["case_001", "case_002", ...],
        "val":   ["case_010", "case_011", ...],
        "test":  ["case_020", "case_021", ...]
    }
}
```

Volumes are automatically resized to `(1, 128, 128, 128)` and min-max normalized to `[0, 1]` at load time.

---

## Training on PDCAD

**1. Edit `configs/train_pdcad.yaml`** to point to your data:

```yaml
data:
  data_root:   "/path/to/your/pdcad/data"
  labels_path: "/path/to/your/labels.json"
  splits_path: "/path/to/your/splits.json"
```

**2. (Optional) Set which GPU to use:**

```bash
export CUDA_VISIBLE_DEVICES=0
```

**3. Run training with ViT-B (recommended starting point):**

```bash
python train.py --config-name train_pdcad model=anymc3d_vitb_pdcad
```

Or with ViT-L for best performance (requires more VRAM):

```bash
python train.py --config-name train_pdcad model=anymc3d_vitl
```

Any config value can be overridden on the command line:

```bash
# Different fold
python train.py --config-name train_pdcad model=anymc3d_vitb_pdcad data.fold=1

# Larger batch size
python train.py --config-name train_pdcad model=anymc3d_vitb_pdcad data.batch_size=4

# Custom run name
python train.py --config-name train_pdcad model=anymc3d_vitb_pdcad model.run_name=my_run
```

Training logs to [Weights & Biases](https://wandb.ai) under the project `pdcad-anymc3d`. Checkpoints are saved to `checkpoints/<run_name>/`.

---

## Inference

Point the inference script at a completed run directory. It auto-discovers the config and best checkpoint (by val AUROC):

```bash
python inference.py --run_dir checkpoints/PDCAD_anymc3d-vitb14-t1c_CosineLR_LoRAlr_1e-6_headlr_5e-6_200ep
```

By default this evaluates the **test** split. To evaluate a different split:

```bash
python inference.py --run_dir checkpoints/<run_name> --split val
```

To use a specific checkpoint instead of the best:

```bash
python inference.py --run_dir checkpoints/<run_name> --checkpoint epoch=45-val_auroc=0.8123.ckpt
```

To override the data path (e.g. running on a different machine):

```bash
python inference.py --run_dir checkpoints/<run_name> --data_root /new/path/to/data
```

**Outputs** are saved inside the run directory:

| File | Description |
|------|-------------|
| `predictions_test.csv` | Per-case predictions, class probabilities, and confidence |
| `summary_test.csv` | Overall and per-class metrics (AUROC, F1, balanced accuracy) |
| `confusion_matrix_test.png` | Confusion matrix plot |
| `roc_curves_test.png` | Per-class ROC curves |

---

## Key hyperparameters

| Parameter | Default (PDCAD) | Description |
|-----------|----------------|-------------|
| `backbone_name` | `dinov2_vitb14` | DINOv2 variant (`vits14`, `vitb14`, `vitl14`) |
| `lora_rank` | 8 | LoRA rank |
| `lora_alpha` | 16 | LoRA scaling |
| `input_size` | 98 | Slice resize resolution |
| `lora_lr` | 1e-6 | Learning rate for LoRA parameters |
| `head_lr` | 5e-6 | Learning rate for classifier head |
| `max_epochs` | 200 | Maximum training epochs |
| `early_stopping_patience` | 40 | Early stopping patience (monitors val loss) |

---

## Citation

```bibtex
@article{liu2025anymc3d,
  title   = {Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification},
  author  = {Liu et al.},
  journal = {arXiv:2512.12887},
  year    = {2025}
}
```
