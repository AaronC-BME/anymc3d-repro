"""
vjepa2_anymc3d.py
-----------------
AnyMC3D using V-JEPA 2 / V-JEPA 2.1 via torch.hub (Strategy A).

V-JEPA 2.1 weights are not yet on HuggingFace, so we load via torch.hub
from the official Meta repo.  Both V-JEPA 2 and V-JEPA 2.1 use the same
native VisionTransformer class and identical forward API, so one file
covers all variants.

torch.hub entry points
----------------------
V-JEPA 2
  'vjepa2_vit_large'      ViT-L  embed_dim=1024  crop=256
  'vjepa2_vit_huge'       ViT-H  embed_dim=1280  crop=256
  'vjepa2_vit_giant'      ViT-g  embed_dim=1408  crop=256
  'vjepa2_vit_giant_384'  ViT-g  embed_dim=1408  crop=384

V-JEPA 2.1
  'vjepa2_1_vit_base_384'      ViT-B  embed_dim=768   crop=384  ← default
  'vjepa2_1_vit_large_384'     ViT-L  embed_dim=1024  crop=384
  'vjepa2_1_vit_giant_384'     ViT-g  embed_dim=1408  crop=384
  'vjepa2_1_vit_gigantic_384'  ViT-G  embed_dim=1536  crop=384

Key differences vs HuggingFace API
------------------------------------
Attribute   torch.hub native         HuggingFace
----------  -----------------------  ---------------------
embed dim   encoder.embed_dim        encoder.config.hidden_size
crop size   inferred from hub name   encoder.config.crop_size
forward     encoder(clip)            encoder(pixel_values_videos=clip, ...)
output      (B, N_tokens, embed_dim) out.last_hidden_state
LoRA names  "qkv", "proj"            "query","key","value","dense"

The hub model forward takes (B, T, 3, H, W) and returns
(B, N_tokens, embed_dim) directly — no wrapper object.

crop_size per hub entry point
------------------------------
All *_384 models  → crop_size = 384
All others        → crop_size = 256
(Parsed from hub_name automatically — no manual setting needed.)

Token count formula
-------------------
N_tokens = (num_frames / tubelet_size) × (crop_size / patch_size)²
  ViT-B 2.1, T=32, crop=384:  (32/2)×(384/16)² = 16×576 = 9,216
  ViT-L 2.0, T=32, crop=256:  (32/2)×(256/16)² = 16×256 = 4,096

Memory guide (RTX 6000 Ada, 49 GB VRAM)
-----------------------------------------
  ViT-B 2.1, T=32, batch=2  →  fits comfortably
  ViT-B 2.1, T=70, batch=1  →  fits (full PDCAD slice coverage)
  ViT-L 2.1, T=32, batch=1  →  fits
  ViT-L 2.1, T=70, batch=1  →  tight; enable gradient checkpointing
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from peft import LoraConfig, get_peft_model
from torchmetrics import AUROC, Accuracy, F1Score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from balanced_accuracy import BalancedAccuracy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])

# crop_size lookup — inferred from the hub entry-point name so users never
# have to set it manually and can't accidentally mismatch it.
_CROP_SIZE: dict[str, int] = {
    # V-JEPA 2
    "vjepa2_vit_large":           256,
    "vjepa2_vit_huge":            256,
    "vjepa2_vit_giant":           256,
    "vjepa2_vit_giant_384":       384,
    # V-JEPA 2.1
    "vjepa2_1_vit_base_384":      384,
    "vjepa2_1_vit_large_384":     384,
    "vjepa2_1_vit_giant_384":     384,
    "vjepa2_1_vit_gigantic_384":  384,
}

_PATCH_SIZE   = 16   # constant across all V-JEPA 2 / 2.1 models
_TUBELET_SIZE = 2    # constant across all V-JEPA 2 / 2.1 models


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """Normalize (N, 3, H, W) float32 in [0,1] with V-JEPA 2 stats."""
    mean = _MEAN.to(x.device, x.dtype).view(1, 3, 1, 1)
    std  = _STD.to(x.device, x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Slice sampler
# ---------------------------------------------------------------------------

def _sample_slices(x: torch.Tensor, num_frames: int, slice_axis: int) -> torch.Tensor:
    """
    Uniformly sample `num_frames` slices from a 5-D volume along `slice_axis`.

    Args:
        x          : (B, C, H, W, S)
        num_frames : must be even  (tubelet_size = 2)
        slice_axis : 1=H (dim 2), 2=W (dim 3), 3=S (dim 4)
                     Use 3 for PDCAD (70 slices along S).
    Returns:
        (B, T, C, h, w)
    """
    assert num_frames % 2 == 0, \
        f"num_frames must be even (tubelet_size=2), got {num_frames}"

    ax       = {1: 2, 2: 3, 3: 4}[slice_axis]
    n_slices = x.shape[ax]
    idx      = torch.linspace(0, n_slices - 1, num_frames).long().to(x.device)
    x        = x.index_select(ax, idx)

    # Move slice axis to dim 1 → (B, T, C, h, w)
    dims = list(range(5))
    dims.remove(ax)
    dims.insert(1, ax)
    return x.permute(*dims)


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class VJEPA2AnyMC3D(nn.Module):
    """
    AnyMC3D built on a frozen + LoRA-adapted V-JEPA 2 / 2.1 encoder,
    loaded via torch.hub from the official Meta repo.
    """

    def __init__(
        self,
        num_classes:   int   = 2,
        hub_name:      str   = "vjepa2_1_vit_base_384",
        lora_rank:     int   = 8,
        lora_alpha:    int   = 16,
        num_frames:    int   = 32,   # must be even (tubelet_size=2)
        slice_axis:    int   = 3,    # 1=H, 2=W, 3=S  — use 3 for PDCAD
        dropout:       float = 0.1,
    ):
        super().__init__()

        assert num_frames % 2 == 0, \
            f"num_frames must be even (tubelet_size=2), got {num_frames}"
        assert hub_name in _CROP_SIZE, \
            f"Unknown hub_name '{hub_name}'. Known: {list(_CROP_SIZE)}"

        self.num_frames  = num_frames
        self.slice_axis  = slice_axis
        self.num_classes = num_classes

        # crop_size is fully determined by hub_name — no manual config needed
        self.crop_size = _CROP_SIZE[hub_name]

        # ── Load encoder via torch.hub and freeze ─────────────────────────────
        print(f"Loading V-JEPA 2 encoder via torch.hub: '{hub_name}' ...")

        # Build model architecture without downloading weights (pretrained=False)
        # The hub entry point returns (encoder, preprocessor) — unpack accordingly
        hub_out = torch.hub.load("facebookresearch/vjepa2", hub_name,
                                 pretrained=False)
        encoder = hub_out[0] if isinstance(hub_out, (tuple, list)) else hub_out

        # Load weights from local checkpoint — weights_only=True avoids the
        # FutureWarning and is safe for checkpoints from a trusted source
        ckpt_path = (
            "/home/jma/Documents/projects/aaron/AnyMC3D"
            "/vjepa_2_1_checkpoint/vjepa2_1_vitb_dist_vitG_384.pt"
        )
        print(f"  Loading weights from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu",
                                weights_only=True)

        # Checkpoint may store weights under an "encoder" key (standard for
        # V-JEPA 2 .pt files); strip common DDP prefixes too
        if isinstance(state_dict, dict) and "encoder" in state_dict:
            state_dict = state_dict["encoder"]
        state_dict = {
            k.replace("module.", "").replace("backbone.", ""): v
            for k, v in state_dict.items()
        }
        msg = encoder.load_state_dict(state_dict, strict=False)
        print(f"  Weights loaded — missing: {len(msg.missing_keys)}, "
              f"unexpected: {len(msg.unexpected_keys)}")

        for p in encoder.parameters():
            p.requires_grad_(False)

        # torch.hub native VisionTransformer exposes embed_dim directly
        self.embed_dim = encoder.embed_dim

        n_tokens = (num_frames // _TUBELET_SIZE) * (self.crop_size // _PATCH_SIZE) ** 2
        print(f"  embed_dim    : {self.embed_dim}")
        print(f"  crop_size    : {self.crop_size}  (from hub_name)")
        print(f"  num_frames   : {num_frames}  (slice_axis={slice_axis})")
        print(f"  tokens/sample: {n_tokens}  "
              f"= ({num_frames}/{_TUBELET_SIZE}) × ({self.crop_size}/{_PATCH_SIZE})²")

        # ── Inject LoRA ────────────────────────────────────────────────────────
        # The torch.hub VisionTransformer uses the same QKV fused projection as
        # DINOv2:
        #   blocks.N.attn.qkv   — fused Q/K/V projection
        #   blocks.N.attn.proj  — output projection
        lora_cfg = LoraConfig(
            r              = lora_rank,
            lora_alpha     = lora_alpha,
            target_modules = ["qkv", "proj"],
            lora_dropout   = 0.0,
            bias           = "none",
        )
        self.encoder = get_peft_model(encoder, lora_cfg)
        self.encoder.print_trainable_parameters()

        # ── Classification head ────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total trainable params: {n_trainable:,}\n")

    # ------------------------------------------------------------------

    def _prepare_clip(self, x: torch.Tensor) -> torch.Tensor:
        """
        (B, 1, H, W, S)  →  (B, 3, T, crop_size, crop_size)

        V-JEPA 2.1 patch_embed is a Conv3d with weight
        [embed_dim, 3, tubelet_size, patch_size, patch_size], so it
        expects channels-first with time in dim 2: (B, C, T, H, W).
        """
        B = x.shape[0]

        # Sample T slices → (B, T, 1, h, w)
        clip = _sample_slices(x, self.num_frames, self.slice_axis)
        T, h, w = clip.shape[1], clip.shape[3], clip.shape[4]

        # Flatten B×T for spatial ops → (B*T, 1, h, w)
        flat = clip.reshape(B * T, 1, h, w)

        # Greyscale → 3-channel
        flat = flat.repeat(1, 3, 1, 1)

        # Resize to crop_size
        if h != self.crop_size or w != self.crop_size:
            flat = F.interpolate(
                flat,
                size=(self.crop_size, self.crop_size),
                mode="bilinear",
                align_corners=False,
            )

        # Clamp and normalise
        flat = flat.clamp(0.0, 1.0)
        flat = _normalize(flat)

        # Restore (B, T, 3, H, W) then permute → (B, 3, T, H, W)
        # V-JEPA 2.1 patch_embed is Conv3d expecting channels-first: (B, C, T, H, W)
        clip = flat.reshape(B, T, 3, self.crop_size, self.crop_size)
        return clip.permute(0, 2, 1, 3, 4).contiguous()  # (B, 3, T, crop_size, crop_size)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 1, H, W, S)  float32, values in [0, 1]
        Returns:
            logits : (B, num_classes)
        """
        # print(f"Shape of input x: {tuple(x.shape)}")
        clip   = self._prepare_clip(x)          # (B, 3, T, crop_size, crop_size)

        # torch.hub VisionTransformer forward:
        #   input  : (B, 3, T, H, W)  ← channels-first, time in dim 2
        #   output : (B, N_tokens, embed_dim)
        tokens = self.encoder(clip)              # (B, N_tokens, embed_dim)

        # Mean-pool all spatiotemporal tokens → volume embedding
        v = tokens.mean(dim=1)                   # (B, embed_dim)

        return self.head(v)                      # (B, num_classes)


# ---------------------------------------------------------------------------
# PyTorch Lightning module
# ---------------------------------------------------------------------------

class VJEPA2LightningModule(L.LightningModule):
    """
    Lightning wrapper for VJEPA2AnyMC3D.
    Selected in train.py via cfg.model.arch == "vjepa2_anymc3d".
    """

    def __init__(
        self,
        num_classes:        int   = 2,
        hub_name:           str   = "vjepa2_1_vit_base_384",
        lora_rank:          int   = 8,
        lora_alpha:         int   = 16,
        num_frames:         int   = 32,
        slice_axis:         int   = 3,
        dropout:            float = 0.1,
        # Optimizer
        lora_lr:            float = 1e-4,
        lora_weight_decay:  float = 1e-5,
        head_lr:            float = 1e-3,
        head_weight_decay:  float = 1e-4,
        warmup_epochs:      int   = 10,
        max_epochs:         int   = 150,
        lr_scheduler:       str   = "cosine",
        # Focal loss
        focal_gamma:        float = 2.0,
        focal_alpha:        float = 0.25,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = VJEPA2AnyMC3D(
            num_classes = num_classes,
            hub_name    = hub_name,
            lora_rank   = lora_rank,
            lora_alpha  = lora_alpha,
            num_frames  = num_frames,
            slice_axis  = slice_axis,
            dropout     = dropout,
        )

        self.num_classes = num_classes

        if num_classes == 2:
            metric_kwargs = dict(task="binary")
        else:
            metric_kwargs = dict(task="multiclass", num_classes=num_classes)

        self.val_auroc   = AUROC(**metric_kwargs)
        self.val_acc     = Accuracy(**metric_kwargs)
        self.val_f1      = F1Score(average="macro", **metric_kwargs)
        self.val_bal_acc = BalancedAccuracy(num_classes=num_classes)

        self.test_auroc   = AUROC(**metric_kwargs)
        self.test_acc     = Accuracy(**metric_kwargs)
        self.test_f1      = F1Score(average="macro", **metric_kwargs)
        self.test_bal_acc = BalancedAccuracy(num_classes=num_classes)

        self.test_preds  = []
        self.test_labels = []

    # ------------------------------------------------------------------

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.hparams.focal_alpha * (1 - pt) ** self.hparams.focal_gamma * ce).mean()

    def _unpack_batch(self, batch):
        x, y, *_ = batch
        if isinstance(x, dict):
            x = next(iter(x.values()))
        return x, y

    def _shared_step(self, batch):
        x, y   = self._unpack_batch(batch)
        logits = self.model(x)
        loss   = self.focal_loss(logits, y)
        probs  = F.softmax(logits, dim=1)
        return loss, probs, y

    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch)
        preds = probs.argmax(dim=1)
        auroc_input = probs[:, 1] if self.num_classes == 2 else probs
        self.val_auroc.update(auroc_input, y)
        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)
        self.val_bal_acc.update(preds, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        self.log("val/AUROC",             self.val_auroc.compute(),   prog_bar=True, sync_dist=True)
        self.log("val/accuracy",          self.val_acc.compute(),     prog_bar=True, sync_dist=True)
        self.log("val/F1_macro",          self.val_f1.compute(),      prog_bar=True, sync_dist=True)
        self.log("val/balanced_accuracy", self.val_bal_acc.compute(), prog_bar=True, sync_dist=True)
        self.val_auroc.reset(); self.val_acc.reset()
        self.val_f1.reset();    self.val_bal_acc.reset()

    def test_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch)
        preds = probs.argmax(dim=1)
        auroc_input = probs[:, 1] if self.num_classes == 2 else probs
        self.test_auroc.update(auroc_input, y)
        self.test_acc.update(preds, y)
        self.test_f1.update(preds, y)
        self.test_bal_acc.update(preds, y)
        self.test_preds.append(probs.cpu())
        self.test_labels.append(y.cpu())

    def on_test_epoch_end(self):
        auroc        = self.test_auroc.compute()
        acc          = self.test_acc.compute()
        f1           = self.test_f1.compute()
        balanced_acc = self.test_bal_acc.compute()

        self.log("test/AUROC",             auroc)
        self.log("test/accuracy",          acc)
        self.log("test/F1_macro",          f1)
        self.log("test/balanced_accuracy", balanced_acc)

        print(f"\n{'='*50}")
        print(f"Test Results:")
        print(f"  AUROC:             {auroc:.4f}")
        print(f"  Accuracy:          {acc:.4f}")
        print(f"  Balanced Accuracy: {balanced_acc:.4f}")
        print(f"  F1 (macro):        {f1:.4f}")
        print(f"{'='*50}\n")

        self.test_auroc.reset(); self.test_acc.reset()
        self.test_f1.reset();    self.test_bal_acc.reset()

    def predict_step(self, batch, batch_idx):
        x, y   = self._unpack_batch(batch)
        logits = self.model(x)
        return y, logits

    # ------------------------------------------------------------------

    def configure_optimizers(self):
        lora_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "head" in name:
                head_params.append(param)
            else:
                lora_params.append(param)

        optimizer = torch.optim.AdamW([
            {"params": lora_params, "lr": self.hparams.lora_lr,
             "weight_decay": self.hparams.lora_weight_decay},
            {"params": head_params, "lr": self.hparams.head_lr,
             "weight_decay": self.hparams.head_weight_decay},
        ])

        if self.hparams.lr_scheduler == "constant":
            return optimizer

        schedulers = []
        milestones  = []

        if self.hparams.warmup_epochs > 0:
            schedulers.append(
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                         total_iters=self.hparams.warmup_epochs)
            )
            milestones.append(self.hparams.warmup_epochs)

        schedulers.append(
            CosineAnnealingLR(
                optimizer,
                T_max   = self.hparams.max_epochs - self.hparams.warmup_epochs,
            )
        )

        scheduler = (
            schedulers[0] if len(schedulers) == 1
            else SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)
        )

        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


# ---------------------------------------------------------------------------
# Sanity check:  python vjepa2_anymc3d.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running VJEPA2AnyMC3D sanity check...\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    model = VJEPA2AnyMC3D(
        num_classes = 2,
        hub_name    = "vjepa2_1_vit_base_384",
        lora_rank   = 8,
        lora_alpha  = 16,
        num_frames  = 8,    # small for fast test; must be even
        slice_axis  = 3,
    ).to(device)

    # PDCAD-like volume — small spatial dims to keep the check fast
    x = torch.rand(2, 1, 384, 384, 70).to(device)

    with torch.no_grad():
        logits = model(x)

    print(f"\nInput  : {tuple(x.shape)}")
    print(f"Output : {tuple(logits.shape)}")   # expected (2, 2)
    assert logits.shape == (2, 2), f"Unexpected shape: {logits.shape}"
    print("\nSanity check passed!")