# AnyMC3D

Scalable 3D Medical Image Classifier adapted from 2D Foundation Models.

Based on: *"Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification"* — Liu et al., 2025 ([arXiv:2512.12887](https://arxiv.org/abs/2512.12887))

This implementation applies AnyMC3D to binary classification on the **PDCAD** dataset using Nuclear Medicine (NM) volumes, and 4-class molecular subtype classification on the **Meningioma T1c** dataset.

---

## How it works

AnyMC3D wraps a frozen DINOv2 backbone with lightweight LoRA adapters. For each 3D volume:

1. All 2D slices are extracted along the chosen axis (axial by default)
2. Each slice is encoded with the LoRA-adapted DINOv2 (ViT-B/14 or ViT-L/14)
3. An attention pooling module aggregates slice embeddings into a single volume embedding
4. A linear head produces the final class prediction

An alternative backbone using **V-JEPA 2.1** (Meta's spatiotemporal ViT-B) is also in development, treating the slice axis natively as a temporal dimension via 3D-RoPE.

> ⚠️ **V-JEPA 2.1 integration is a work in progress.** `model_arch/vjepa2_anymc3d.py` and `configs/model/vjepa21_vitb.yaml` are not yet fully refactored and should not be used for training.

---

## Project structure

```
AnyMC3D/
├── train.py                        # Hydra-driven training script
├── inference_online.py             # Inference + metrics + plots (TODO: update for refactor)
├── inference_nifti.py              # Inference directly from raw NIfTI files
├── balanced_accuracy.py            # Custom balanced accuracy metric
├── preprocess.py                   # Preprocessing pipeline (MONAI percentile normalization)
├── pyproject.toml                  # Project dependencies (uv / pip)
├── uv.lock                         # Locked dependency versions
│
├── model_arch/
│   ├── __init__.py
│   ├── anymc3d.py                  # DINOv2-based model + Lightning module
│   └── vjepa2_anymc3d.py          # V-JEPA 2.1-based model + Lightning module
│
├── data_modules/
│   ├── __init__.py
│   ├── pdcad_dataset.py            # PDCAD Dataset and DataModule
│   └── data_augmentation.py        # MONAI augmentation pipeline (nnU-Net style)
│
├── configs/
│   ├── train.yaml                  # Top-level training config (defaults list)
│   ├── data/
│   │   └── pdcad.yaml              # PDCAD data config
│   └── model/
│       ├── anymc3d_vitb.yaml       # DINOv2 ViT-B/14 model config for PDCAD
│       └── vjepa21_vitb.yaml       # V-JEPA 2.1 ViT-B model config for PDCAD
│
└── checkpoints/                    # Model checkpoints (per run, gitignored)
```

---

## Dataset Download

The PDCAD dataset (Nuclear Medicine (NM) volumes, labels, and splits) is available for download via Google Drive:

> 📁 **[Download PDCAD dataset](https://drive.google.com/drive/folders/1cLZosBVq2HbyDH0BbSxf90J4M96A_4gi?usp=drive_link)**

The archive includes:
- `labels.json` — binary labels for all cases
- `splits.json` — train/val/test splits
- Pre-processed `.npy` volumes (float32, channel-first, ready for training)

After downloading, point `data_root`, `labels_path`, and `splits_path` in `configs/data/pdcad.yaml` to the extracted directory.

> **Note on preprocessing:** The `.npy` files are pre-converted from the original `.nii.gz` images using `preprocess.py`. If you need to re-run preprocessing from raw NIfTI files (e.g. to change normalization parameters), see the [Preprocessing](#preprocessing) section below.

---

## Finetuned Model

A finetuned ViT-B/14 checkpoint on the PDCAD dataset is available on Hugging Face:

> 🤗 **[aaronchoi6/anymc3d-vitb14-pdcad](https://huggingface.co/aaronchoi6/anymc3d-vitb14-pdcad)**

| Detail | Value |
|--------|-------|
| Backbone | DINOv2 ViT-B/14 (frozen) |
| Trainable params | ~455K (LoRA + classifier head) |
| Best val AUROC | 0.90 |
| Training | 150 epochs, fold 0 |

---

## Installation

**Option A: Using uv (recommended)**

```bash
uv sync
```

This creates a `.venv`, resolves all dependencies (including PyTorch with CUDA 12.1), and installs everything in one step.

Activate the environment:
```bash
source .venv/bin/activate
```

**Option B: Using pip**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

---

## Preprocessing

Raw NIfTI volumes are preprocessed with `preprocess.py` before training. The pipeline:

1. Load raw `.nii.gz` and reorient to RAS+ canonical orientation
2. Normalize intensities using **MONAI `ScaleIntensityRangePercentiles`**:
   ```
   lower=0, upper=99.5, b_min=0, b_max=1, clip=True, relative=False
   ```
3. Reshape to `(1, H, W, S)` channel-first for PyTorch
4. Save as float32 `.npy`

To run preprocessing on raw NIfTI files:

```bash
python preprocess.py \
    --input_dir  /path/to/raw/nifti/folder \
    --output_dir /path/to/output/folder \
    --workers    4 \
    --verify
```

For PDCAD, no spatial resizing is applied — volumes preserve their original shape `(1, 300, 300, 70)`.

---

## PDCAD dataset format

The dataset loader expects a **flat** directory of preprocessed `.npy` volumes:
```
<data_root>/
    <case_id>_0000.npy     # Preprocessed NM volume (1, H, W, S) float32, values in [0, 1]
    ...
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

**`splits.json`** — defines the train/val/test splits per fold:
```json
{
    "0": {
        "train": ["case_001", "case_002", ...],
        "val":   ["case_010", "case_011", ...],
        "test":  ["case_020", "case_021", ...]
    }
}
```

---

## Training

### Configuration

All hyperparameters — including optimizer, loss, scheduler, and WandB settings — live in the model config YAML. The top-level `configs/train.yaml` just specifies which data and model configs to use via its defaults list.

Data paths are set in `configs/data/pdcad.yaml`:
```yaml
module:
  data_root:   "/path/to/your/pdcad/data"
  labels_path: "/path/to/your/labels.json"
  splits_path: "/path/to/your/splits.json"
```

### Running training

```bash
# (Optional) set which GPU to use
export CUDA_VISIBLE_DEVICES=0

# DINOv2 ViT-B/14 on PDCAD (recommended starting point)
python train.py data=pdcad model=anymc3d_vitb

# V-JEPA 2.1 ViT-B on PDCAD (TODO: not yet ready — refactor in progress)
# python train.py data=pdcad model=vjepa21_vitb

# Background with logging
nohup python train.py data=pdcad model=anymc3d_vitb > logs/anymc3d_vitb_pdcad.log 2>&1 &
```

Any config value can be overridden on the command line:

```bash
# Different fold
python train.py data=pdcad model=anymc3d_vitb data.module.fold=1

# Larger batch size
python train.py data=pdcad model=anymc3d_vitb data.module.batch_size=4

# Multi-fold
python train.py data=pdcad model=anymc3d_vitb 'data.module.fold=[0,1,2]'

# Override a model hyperparameter
python train.py data=pdcad model=anymc3d_vitb model.lora_lr=2e-4
```

Training logs to [Weights & Biases](https://wandb.ai) under the project specified in the model config (`pdcad-anymc3d` by default). Checkpoints are saved under `checkpoints/<run_name>/`.

---

## Inference

> ⚠️ `inference.py` is pending updates for the refactored codebase.

---

## Key hyperparameters

| Parameter | Default (PDCAD ViT-B) | Description |
|-----------|----------------------|-------------|
| `backbone_name` | `dinov2_vitb14` | DINOv2 variant (`vits14`, `vitb14`, `vitl14`) |
| `lora_rank` | 8 | LoRA rank |
| `lora_alpha` | 16 | LoRA scaling |
| `input_size` | 308 | Slice resize resolution fed to DINOv2 (must be a multiple of 14) |
| `slice_axis` | 3 | Axis to extract 2D slices along (axis 3 = 70 slices for PDCAD) |
| `vision_blocks` | 0 | Number of learnable transformer blocks on top of backbone (0 = disabled) |
| `lora_lr` | 1e-4 | Learning rate for LoRA parameters |
| `head_lr` | 5e-4 | Learning rate for classifier head |
| `lr_scheduler` | `cosine` | LR schedule (`cosine` or `constant`) |
| `focal_gamma` | 2.0 | Focal loss focusing parameter |
| `focal_alpha` | 0.5 | Focal loss class balancing (0.5 for balanced datasets) |
| `max_epochs` | 150 | Maximum training epochs |
| `precision` | `16-mixed` | Mixed precision training |
| `batch_size` | 2 | Training batch size (set in data config) |
| `patch_size` | `[300, 300, 70]` | Input volume dimensions (set in data config) |

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