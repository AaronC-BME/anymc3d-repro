# AnyMC3D

Scalable 3D Medical Image Classifier adapted from 2D Foundation Models.

Based on: *"Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification"* — Liu et al., 2025 ([arXiv:2512.12887](https://arxiv.org/abs/2512.12887))

This implementation applies AnyMC3D to binary classification on the **PDCAD** dataset using Nuclear Medicine (NM) volumes.

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
├── pyproject.toml          # Project dependencies (uv / pip)
├── configs/
│   ├── train_pdcad.yaml    # Top-level PDCAD training config
│   └── model/
│       ├── anymc3d_vitb_pdcad.yaml   # ViT-B model config for PDCAD
│       └── anymc3d_vitl.yaml         # ViT-L model config
└── outputs/                # All training and inference outputs
    ├── checkpoints/        # Model checkpoints (per run)
    ├── predictions/        # Inference CSVs and plots
    └── logs/               # W&B local logs and Hydra outputs
```

---

## Dataset Download

The PDCAD dataset (Nuclear Medicine (NM) volumes, labels, and splits) is available for download via Google Drive:

> 📁 **[Download PDCAD dataset](https://drive.google.com/drive/folders/1cLZosBVq2HbyDH0BbSxf90J4M96A_4gi?usp=drive_link)**

The archive includes:
- `labels.json` — binary labels for all cases
- `splits.json` — train/val split (200 training cases, 100 validation cases)
- Pre-processed `.b2nd` volumes (Blosc2-compressed, ready for training — no conversion needed)

After downloading, extract the archive and point `data_root` in `configs/train_pdcad.yaml` to the extracted directory.

> **Note on preprocessing:** The `.b2nd` files are pre-converted from the original `.nii.gz` images using [nnssl](https://github.com/MIC-DKFZ/nnssl). If you need to re-run preprocessing from raw NIfTI files (e.g., to change the voxel spacing or crop parameters), a conversion script is available — open an issue or contact the maintainers.

---

## Installation

**Option A: Using uv (recommended)**

```bash
uv sync
```

This creates a `.venv`, resolves all dependencies (including PyTorch with CUDA 12.1), and installs everything in one step.

To run scripts:
```bash
uv run python train.py --config-name train_pdcad model=anymc3d_vitb_pdcad
```

Or activate the environment manually:
```bash
source .venv/bin/activate
python train.py --config-name train_pdcad model=anymc3d_vitb_pdcad
```

**Option B: Using pip**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

---

## PDCAD dataset format

The dataset loader expects the following directory structure:

```
<data_root>/
    <case_id>/
        ses-DEFAULT/
            <case_id>_0000.b2nd     # Blosc2-compressed NM volume (1, D, H, W) float32
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

**`splits.json`** — defines the train/val split:
```json
{
    "0": {
        "train": ["case_001", "case_002", ...],
        "val":   ["case_010", "case_011", ...]
    }
}
```

Volumes are automatically resized to `(1, 128, 128, 128)` and min-max normalized to `[0, 1]` at load time.

---

## Training on PDCAD

> ⚠️ **Before training**, update the dataset paths in `configs/train_pdcad.yaml` to point to your local data:

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

Training logs to [Weights & Biases](https://wandb.ai) under the project `pdcad-anymc3d`. All outputs are saved under the `outputs/` directory:

- **Checkpoints** → `outputs/checkpoints/<run_name>/`
- **Loss curves and training logs** → `outputs/logs/`

---

## Inference

Point the inference script at a completed run directory. It auto-discovers the config and best checkpoint (by val AUROC):

```bash
python inference.py --run_dir outputs/checkpoints/PDCAD_anymc3d-vitb14-t1c_CosineLR_LoRAlr_1e-6_headlr_5e-6_200ep
```

By default this evaluates the **test** split. To evaluate a different split:

```bash
python inference.py --run_dir outputs/checkpoints/<run_name> --split val
```

To use a specific checkpoint instead of the best:

```bash
python inference.py --run_dir outputs/checkpoints/<run_name> --checkpoint epoch=45-val_auroc=0.8123.ckpt
```

To override the data path (e.g. running on a different machine):

```bash
python inference.py --run_dir outputs/checkpoints/<run_name> --data_root /new/path/to/data
```

**Outputs** are saved inside `outputs/predictions/<run_name>/`:

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
| `input_size` | 308 | Slice resize resolution |
| `lora_lr` | 1e-4 | Learning rate for LoRA parameters |
| `head_lr` | 1e-3 | Learning rate for classifier head |
| `max_epochs` | 150 | Maximum training epochs |
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