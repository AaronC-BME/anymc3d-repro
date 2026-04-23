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

Token count formula
-------------------
N_tokens = (num_frames / tubelet_size) × (crop_size / patch_size)²
         = T' × (H' × W')
  where T'      = time-tubes   (each covers 2 consecutive sampled slices)
        H' × W' = spatial grid (per time-tube)

Each "time-tube" in V-JEPA 2.1 plays the role of a "slice" in the DINOv2
version of AnyMC3D — it is the finest temporal unit available after tubelet
embedding.  All pooling flags below operate on this (T', H'×W') factorisation.

AnyMC3D-parity flags
--------------------
V-JEPA has no CLS token, so flag semantics differ slightly from the DINOv2
version.  Sensible defaults preserve backward compatibility with existing
V-JEPA checkpoints (all False → flat mean-pool over all tokens).

  use_25d (bool)
    True  : each sampled slice becomes [s-1, s, s+1] stacked along the channel
            axis (neighbours from ORIGINAL slice space, with boundary padding).
    False : single slice replicated to 3 channels (original behaviour).

  use_patch_attn_pool (bool)
    True  : AttentionPool over the H'×W' spatial tokens of each time-tube,
            yielding a per-time-tube feature of dim D.
    False : mean-pool over H'×W' (per-time-tube, dim D).

  use_patch_concat (bool)
    True  : per-time-tube feature = [attn_pool(spatial) ; mean(spatial)] → 2D.
            This is the V-JEPA analogue of DINOv2's [CLS ; mean(patches)] —
            two complementary views of the spatial tokens concatenated.
            Implies an internal AttentionPool (no need to also set
            use_patch_attn_pool).
    False : per-time-tube feature dim = D (governed by use_patch_attn_pool).

  use_slice_attn_pool (bool)
    True  : AttentionPool over T' time-tubes → volume embedding.
    False : mean over T' time-tubes → volume embedding (original behaviour).

Checkpoint loading behaviour
-----------------------------
`vjepa_checkpoint_path` controls whether pretrained base weights are loaded
during __init__:

  - Training:   pass the explicit path to the .pt file
  - Inference:  pass None (default) — Lightning's load_from_checkpoint
                restores the full model state from the .ckpt; loading base
                weights here would be redundant and may fail on other machines.

Old checkpoints (saved before these parameters were added) are safe: they did
not serialise the new hparams, so Lightning will use the defaults (all False /
None) and the resulting module will be byte-for-byte compatible with the
pre-refactor mean-pool architecture.
"""

from __future__ import annotations

import warnings
from pathlib import Path

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
# Slice samplers
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


def _sample_slices_25d(x: torch.Tensor, num_frames: int, slice_axis: int) -> torch.Tensor:
    """
    Sample `num_frames` centre slices and form 2.5D triplets using ORIGINAL-space
    neighbours [s-1, s, s+1] stacked along the channel axis.  Boundary slices are
    padded by repeating the first/last original slice.

    Args:
        x          : (B, C, H, W, S)   C=1 for single-channel MRI/NM
        num_frames : must be even
        slice_axis : 1=H, 2=W, 3=S

    Returns:
        (B, T, 3, h, w)   ready to feed to the 2D→3D patch_embed as RGB channels
    """
    assert num_frames % 2 == 0, \
        f"num_frames must be even (tubelet_size=2), got {num_frames}"

    ax       = {1: 2, 2: 3, 3: 4}[slice_axis]
    n_slices = x.shape[ax]

    centre = torch.linspace(0, n_slices - 1, num_frames).long().to(x.device)
    prev_  = (centre - 1).clamp(min=0)
    next_  = (centre + 1).clamp(max=n_slices - 1)

    p = x.index_select(ax, prev_)   # same shape as x but with T along ax
    c = x.index_select(ax, centre)
    n = x.index_select(ax, next_)

    # Move T-axis → dim 1 so each tensor is (B, T, C, h, w)
    dims = list(range(5))
    dims.remove(ax)
    dims.insert(1, ax)
    p, c, n = p.permute(*dims), c.permute(*dims), n.permute(*dims)

    # Concat along channel dim → (B, T, 3C, h, w).  C=1 → 3 channels = RGB-like.
    return torch.cat([p, c, n], dim=2)


# ---------------------------------------------------------------------------
# Attention pooling
# ---------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """
    Query-based attention pool — mirrors the AttentionPool in anymc3d.py.
    Aggregates a sequence [B, L, D] into [B, D] using a single learnable query.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            H : (B, L, D)
        Returns:
            v : (B, D)
            a : (B, L)   attention weights
        """
        scale  = H.shape[-1] ** 0.5
        scores = torch.einsum("bld,d->bl", H, self.query) / scale
        a      = F.softmax(scores, dim=-1)
        v      = torch.einsum("bl,bld->bd", a, H)
        return v, a


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
        num_classes:           int        = 2,
        hub_name:              str        = "vjepa2_1_vit_base_384",
        lora_rank:             int        = 8,
        lora_alpha:            int        = 16,
        num_frames:            int        = 32,   # must be even (tubelet_size=2)
        slice_axis:            int        = 3,    # 1=H, 2=W, 3=S  — use 3 for PDCAD
        dropout:               float      = 0.1,
        vjepa_checkpoint_path: str | None = None,
        # ── AnyMC3D-parity flags (see module docstring) ──────────────────────
        use_slice_attn_pool:   bool       = False,
        use_patch_attn_pool:   bool       = False,
        use_patch_concat:      bool       = False,
        use_25d:               bool       = False,
        
    ):
        super().__init__()

        assert num_frames % 2 == 0, \
            f"num_frames must be even (tubelet_size=2), got {num_frames}"
        assert hub_name in _CROP_SIZE, \
            f"Unknown hub_name '{hub_name}'. Known: {list(_CROP_SIZE)}"

        self.num_frames          = num_frames
        self.slice_axis          = slice_axis
        self.num_classes         = num_classes
        self.use_slice_attn_pool = use_slice_attn_pool
        self.use_patch_attn_pool = use_patch_attn_pool
        self.use_patch_concat    = use_patch_concat
        self.use_25d             = use_25d

        # crop_size is fully determined by hub_name — no manual config needed
        self.crop_size = _CROP_SIZE[hub_name]

        # Precompute the spatiotemporal grid shape so forward() can reshape tokens
        self.t_prime  = num_frames // _TUBELET_SIZE
        self.hw_prime = (self.crop_size // _PATCH_SIZE) ** 2

        # ── Load encoder architecture via torch.hub (weights-free) ────────────
        print(f"Loading V-JEPA 2 encoder architecture via torch.hub: '{hub_name}' ...")
        hub_out = torch.hub.load("facebookresearch/vjepa2", hub_name,
                                 pretrained=False)
        encoder = hub_out[0] if isinstance(hub_out, (tuple, list)) else hub_out

        # ── Optionally load pretrained base weights ───────────────────────────
        if vjepa_checkpoint_path is not None:
            ckpt_path = Path(vjepa_checkpoint_path)
            if ckpt_path.exists():
                print(f"  Loading base weights from: {ckpt_path}")
                state_dict = torch.load(str(ckpt_path), map_location="cpu",
                                        weights_only=True)
                if isinstance(state_dict, dict) and "encoder" in state_dict:
                    state_dict = state_dict["encoder"]
                state_dict = {
                    k.replace("module.", "").replace("backbone.", ""): v
                    for k, v in state_dict.items()
                }
                msg = encoder.load_state_dict(state_dict, strict=False)
                print(f"  Weights loaded — missing: {len(msg.missing_keys)}, "
                      f"unexpected: {len(msg.unexpected_keys)}")
            else:
                warnings.warn(
                    f"vjepa_checkpoint_path '{vjepa_checkpoint_path}' not found — "
                    "encoder will start from random weights. "
                    "If loading from a Lightning checkpoint this is expected and fine.",
                    stacklevel=2,
                )
        else:
            print("  vjepa_checkpoint_path=None — skipping base-weight load. "
                  "Lightning will restore weights from the .ckpt file.")

        for p in encoder.parameters():
            p.requires_grad_(False)

        # torch.hub native VisionTransformer exposes embed_dim directly
        self.embed_dim = encoder.embed_dim

        # Per-time-tube feature dim is doubled when use_patch_concat=True
        self.slice_embed_dim = self.embed_dim * 2 if use_patch_concat else self.embed_dim

        n_tokens = self.t_prime * self.hw_prime
        print(f"  embed_dim       : {self.embed_dim}")
        print(f"  crop_size       : {self.crop_size}  (from hub_name)")
        print(f"  num_frames      : {num_frames}  (slice_axis={slice_axis})")
        print(f"  grid            : T'={self.t_prime}, H'·W'={self.hw_prime}, "
              f"N_tokens={n_tokens}")
        print(f"  slice_embed_dim : {self.slice_embed_dim}")
        print(f"  flags           : use_25d={use_25d}, "
              f"patch_attn={use_patch_attn_pool}, patch_concat={use_patch_concat}, "
              f"slice_attn={use_slice_attn_pool}")

        # ── Inject LoRA ────────────────────────────────────────────────────────
        lora_cfg = LoraConfig(
            r              = lora_rank,
            lora_alpha     = lora_alpha,
            target_modules = ["qkv", "proj"],
            lora_dropout   = 0.0,
            bias           = "none",
        )
        self.encoder = get_peft_model(encoder, lora_cfg)
        self.encoder.print_trainable_parameters()

        # ── Optional pooling modules ──────────────────────────────────────────
        # Spatial AttentionPool (per-time-tube over H'·W'):
        # needed if we attn-pool patches directly, OR if we concat attn+mean.
        if use_patch_attn_pool or use_patch_concat:
            self.patch_pool = AttentionPool(self.embed_dim)

        # Time-tube AttentionPool (over T'):
        if use_slice_attn_pool:
            self.slice_pool = AttentionPool(self.slice_embed_dim)

        # ── Classification head ────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.slice_embed_dim, num_classes),
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

        If use_25d: each sampled centre slice is paired with its [s-1, s, s+1]
        neighbours in ORIGINAL-slice space, stacked as 3 channels.
        Otherwise: single slice replicated to 3 channels.
        """
        B = x.shape[0]

        if self.use_25d:
            # (B, T, 3, h, w) — 3 channels already = original neighbours
            clip = _sample_slices_25d(x, self.num_frames, self.slice_axis)
            T, _, h, w = clip.shape[1], clip.shape[2], clip.shape[3], clip.shape[4]
            flat = clip.reshape(B * T, 3, h, w)
        else:
            # (B, T, 1, h, w) — replicate to 3 channels
            clip = _sample_slices(x, self.num_frames, self.slice_axis)
            T, h, w = clip.shape[1], clip.shape[3], clip.shape[4]
            flat = clip.reshape(B * T, 1, h, w).repeat(1, 3, 1, 1)

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

        # (B, T, 3, crop, crop) → (B, 3, T, crop, crop)  — V-JEPA Conv3d layout
        clip = flat.reshape(B, T, 3, self.crop_size, self.crop_size)
        return clip.permute(0, 2, 1, 3, 4).contiguous()

    # ------------------------------------------------------------------

    def _pool_spatial(self, tokens_bthd: torch.Tensor) -> torch.Tensor:
        """
        Spatial aggregation within each time-tube.

        Args:
            tokens_bthd : (B, T', H'·W', D)
        Returns:
            per-time-tube feature: (B, T', slice_embed_dim)
        """
        B, T, HW, D = tokens_bthd.shape

        if self.use_patch_concat:
            # Concat [attn_pool ; mean] → 2D per time-tube
            flat = tokens_bthd.reshape(B * T, HW, D)
            attn_feat, _ = self.patch_pool(flat)         # (B*T, D)
            attn_feat    = attn_feat.reshape(B, T, D)    # (B, T, D)
            mean_feat    = tokens_bthd.mean(dim=2)       # (B, T, D)
            return torch.cat([attn_feat, mean_feat], dim=-1)  # (B, T, 2D)

        if self.use_patch_attn_pool:
            flat = tokens_bthd.reshape(B * T, HW, D)
            out, _ = self.patch_pool(flat)               # (B*T, D)
            return out.reshape(B, T, D)

        # Default: mean-pool over spatial grid
        return tokens_bthd.mean(dim=2)                   # (B, T, D)

    def _pool_slices(self, slice_feat: torch.Tensor) -> torch.Tensor:
        """
        Aggregate per-time-tube features into a single volume embedding.

        Args:
            slice_feat : (B, T', slice_embed_dim)
        Returns:
            (B, slice_embed_dim)
        """
        if self.use_slice_attn_pool:
            v, _ = self.slice_pool(slice_feat)
            return v
        return slice_feat.mean(dim=1)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 1, H, W, S)  float32, values in [0, 1]
        Returns:
            logits : (B, num_classes)
        """
        clip   = self._prepare_clip(x)          # (B, 3, T, crop, crop)

        # torch.hub VisionTransformer forward:
        #   input  : (B, 3, T, H, W)
        #   output : (B, N_tokens, embed_dim)  with N_tokens = T' · H'·W'
        tokens = self.encoder(clip)              # (B, N_tokens, D)

        B, N, D = tokens.shape
        assert N == self.t_prime * self.hw_prime, (
            f"Unexpected token count {N}; expected "
            f"T'={self.t_prime} × H'·W'={self.hw_prime} = {self.t_prime * self.hw_prime}"
        )
        # Reshape to (B, T', H'·W', D) — separate temporal and spatial factors
        tokens = tokens.reshape(B, self.t_prime, self.hw_prime, D)

        # Two-stage pooling: spatial (within time-tube) → temporal (across tubes)
        slice_feat = self._pool_spatial(tokens)           # (B, T', slice_embed_dim)
        v          = self._pool_slices(slice_feat)        # (B, slice_embed_dim)

        return self.head(v)                                # (B, num_classes)


# ---------------------------------------------------------------------------
# PyTorch Lightning module
# ---------------------------------------------------------------------------

class VJEPA2LightningModule(L.LightningModule):
    """
    Lightning wrapper for VJEPA2AnyMC3D.
    Selected via cfg.model._target_ = vjepa2_anymc3d.VJEPA2LightningModule.
    """

    def __init__(
        self,
        num_classes:           int        = 2,
        hub_name:              str        = "vjepa2_1_vit_base_384",
        lora_rank:             int        = 8,
        lora_alpha:            int        = 16,
        num_frames:            int        = 32,
        slice_axis:            int        = 3,
        dropout:               float      = 0.1,
        vjepa_checkpoint_path: str | None = None,
        # ── AnyMC3D-parity flags ──────────────────────────────────────────────
        use_slice_attn_pool:   bool       = False,
        use_patch_attn_pool:   bool       = False,
        use_patch_concat:      bool       = False,
        use_25d:               bool       = False,
        # ── Optimizer ─────────────────────────────────────────────────────────
        lora_lr:               float      = 1e-4,
        lora_weight_decay:     float      = 1e-5,
        head_lr:               float      = 1e-3,
        head_weight_decay:     float      = 1e-4,
        warmup_epochs:         int        = 10,
        max_epochs:            int        = 150,
        lr_scheduler:          str        = "cosine",
        # ── Focal loss ────────────────────────────────────────────────────────
        focal_gamma:           float      = 2.0,
        focal_alpha:           float      = 0.25,
        # Training hyperparameters
        seed:                 int   = 0,
        precision:            str   = "16-mixed",
        early_stopping_patience: int = 50,
        save_top_k:           int   = 3,
        log_every_n_steps:    int   = 10,
        #WandB logging
        project:              str   = "AnyMC3D",
        run_name:             str   = "VJEPA2_1_AnyMC3D_Run",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = VJEPA2AnyMC3D(
            num_classes           = num_classes,
            hub_name              = hub_name,
            lora_rank             = lora_rank,
            lora_alpha            = lora_alpha,
            num_frames            = num_frames,
            slice_axis            = slice_axis,
            dropout               = dropout,
            vjepa_checkpoint_path = vjepa_checkpoint_path,
            use_slice_attn_pool   = use_slice_attn_pool,
            use_patch_attn_pool   = use_patch_attn_pool,
            use_patch_concat      = use_patch_concat,
            use_25d               = use_25d,
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
            # Head, patch_pool, slice_pool are all "head-like" (trained from
            # scratch, higher LR) in contrast to LoRA adapters on the frozen
            # encoder.
            if any(key in name for key in ("head", "patch_pool", "slice_pool")):
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
                T_max = self.hparams.max_epochs - self.hparams.warmup_epochs,
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

    # Exercise all feature flags together
    model = VJEPA2AnyMC3D(
        num_classes           = 2,
        hub_name              = "vjepa2_1_vit_base_384",
        lora_rank             = 8,
        lora_alpha            = 16,
        num_frames            = 8,    # small for fast test; must be even
        slice_axis            = 3,
        vjepa_checkpoint_path = None,
        use_slice_attn_pool   = True,
        use_patch_attn_pool   = True,
        use_patch_concat      = True,
        use_25d               = True,
    ).to(device)

    # PDCAD-like volume — small spatial dims to keep the check fast
    x = torch.rand(2, 1, 384, 384, 70).to(device)

    with torch.no_grad():
        logits = model(x)

    print(f"\nInput  : {tuple(x.shape)}")
    print(f"Output : {tuple(logits.shape)}")   # expected (2, 2)
    assert logits.shape == (2, 2), f"Unexpected shape: {logits.shape}"
    print("\nSanity check passed!")