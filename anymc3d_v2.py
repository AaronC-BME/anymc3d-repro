"""
AnyMC3D v2 - Scalable 3D Medical Image Classifier adapted from 2D Foundation Models
Based on: "Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification"
         Liu et al., 2025 (arXiv:2512.12887)

Changes from v1:
  [NEW] 2.5D input: instead of repeating a single slice 3x, stacks [s-1, s, s+1]
        as 3 channels, giving the backbone genuine inter-slice context per forward pass.
  [NEW] Patch attention pooling: instead of average-pooling patch tokens, uses a
        learnable AttentionPool query to aggregate them — matching the slice-level
        fusion strategy and enriching spatial detail in the per-slice representation.
  [NEW] CLS + patch concat: following dino.txt (Jose et al., 2024, arXiv:2412.16334),
        the [CLS] token is concatenated with the pooled patch representation to form
        a 2D-sized slice embedding that captures both global and local information.
        This doubles embed_dim from D -> 2*D throughout the model.

Adapted for Meningioma Molecular Subtype Classification (MG1-MG4)
using T1-contrast and T2-weighted MRI data.

Usage:
    # Single modality
    model = AnyMC3D(num_classes=4, modalities=['t1c'])

    # Multi-modal (T1c + T2w) - recommended for meningioma
    model = AnyMC3D(num_classes=4, modalities=['t1c', 't2w'])
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from peft import LoraConfig, get_peft_model
import lightning as L
from torchmetrics import AUROC, Accuracy, F1Score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from balanced_accuracy import BalancedAccuracy


# ---------------------------------------------------------------
# ImageNet normalization constants (DINOv2 pretraining stats)
# ---------------------------------------------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


def normalize_slices(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of RGB slices with ImageNet statistics.
    Args:
        x: [B, 3, H, W] float tensor in range [0, 1]
    Returns:
        [B, 3, H, W] normalized tensor
    """
    mean = IMAGENET_MEAN.to(x.device).view(1, 3, 1, 1)
    std  = IMAGENET_STD.to(x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------
# Query-based Attention Pooling (Slice Fusion)
# ---------------------------------------------------------------

class AttentionPool(nn.Module):
    """
    Aggregates a sequence of embeddings using a single learnable query vector.
    Permutation-invariant — suits MRI with variable slice counts.
    Implements Eq. 5 from the AnyMC3D paper.

    Used in two places in v2:
      1. Slice-level fusion (same as v1) — aggregates slice embeddings into a volume embedding
      2. [NEW] Patch-level fusion — aggregates patch tokens within each slice into a
         single patch representation (replaces average pooling from v1)
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, H: torch.Tensor):
        """
        Args:
            H: [B, S, D] sequence of embeddings (slices or patches)
        Returns:
            v: [B, D]    aggregated embedding
            a: [B, S]    attention weights
        """
        scale  = H.shape[-1] ** 0.5
        scores = torch.einsum('bsd,d->bs', H, self.query) / scale
        a      = F.softmax(scores, dim=-1)
        v      = torch.einsum('bs,bsd->bd', a, H)
        return v, a


# ---------------------------------------------------------------
# Single-Modality Encoder
# ---------------------------------------------------------------

class ModalityEncoder(nn.Module):
    """
    Encodes a single 3D MRI volume into a fixed-size embedding.
    Slices along the highest-resolution axis only — matching AnyMC3D Section 3.2.

    v2 changes:
      - 2.5D input: [s-1, s, s+1] stacked as 3 channels per forward pass
      - Per-slice embedding = concat([CLS], AttentionPool(patch_tokens)) -> 2*D
      - slice-level pool operates on 2*D embeddings
    """

    def __init__(
        self,
        backbone:    nn.Module,
        embed_dim:   int,
        lora_rank:   int = 8,
        lora_alpha:  int = 16,
        input_size:  int = 224,
        slice_axis:  int = 3,
    ):
        super().__init__()
        # embed_dim here is the RAW backbone embed_dim (e.g. 768 for ViT-B)
        # After CLS+patch concat the actual slice embedding will be 2*embed_dim
        self.embed_dim  = embed_dim
        self.input_size = input_size
        self.slice_axis = slice_axis

        lora_config = LoraConfig(
            r              = lora_rank,
            lora_alpha     = lora_alpha,
            target_modules = ["qkv", "proj", "patch_embed.proj"],
            lora_dropout   = 0.0,
            bias           = "none",
        )
        self.adapted_backbone = get_peft_model(backbone, lora_config)

        # -------------------------------------------------------
        # [NEW] Patch attention pool — one learnable query that
        # aggregates the N patch tokens for each slice into a
        # single D-dimensional vector, replacing average pooling.
        # Operates on the raw backbone embed_dim (D).
        # -------------------------------------------------------
        self.patch_pool = AttentionPool(embed_dim)

        # Slice-level pool operates on the CONCATENATED embedding (2*D)
        # because each slice embedding is now [CLS ; patch_pooled]
        self.pool = AttentionPool(embed_dim * 2)

    def encode_slices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract per-slice embeddings from the adapted backbone.

        v1: returned only the [CLS] token -> shape [B, D]
        [NEW] v2: concatenates [CLS] with attention-pooled patch tokens
                  following dino.txt (Jose et al., 2024) Eq. 2.
                  Returns shape [B, 2*D].
        """
        out = self.adapted_backbone.forward_features(x)
        if isinstance(out, dict):
            cls_token    = out['x_norm_clstoken']       # [B, D]
            patch_tokens = out['x_norm_patchtokens']    # [B, N, D]
        else:
            cls_token    = out[:, 0]                    # [B, D]
            patch_tokens = out[:, 1:]                   # [B, N, D]

        # [NEW] Attention pool over patch tokens instead of average pooling.
        # Each patch token votes on its relevance via a learned query,
        # allowing the model to focus on diagnostically relevant spatial regions.
        patch_pooled, _ = self.patch_pool(patch_tokens)     # [B, D]

        # [NEW] Concatenate CLS (global) with patch_pooled (local) -> [B, 2*D]
        # CLS captures overall slice-level context; patch_pooled captures
        # spatially-distributed local features — together they form a richer
        # per-slice representation for downstream slice fusion.
        return torch.cat([cls_token, patch_pooled], dim=-1)  # [B, 2*D]

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 1, H, W, S]  values in [0, 1]
        Returns:
            v:    [B, 2*D]       modality embedding
            attn: [B, n_slices]  attention weights
        """
        B, C, H, W, S = x.shape

        # Move slice axis to position 1 so we can index along it
        spatial_to_full = {1: 2, 2: 3, 3: 4}
        full_axis = spatial_to_full[self.slice_axis]

        perm     = [0, full_axis] + [i for i in range(1, 5) if i != full_axis]
        x_perm   = x.permute(*perm)          # (B, n_slices, C, h, w)
        n_slices = x_perm.shape[1]
        h_sz     = x_perm.shape[3]
        w_sz     = x_perm.shape[4]

        # -------------------------------------------------------
        # [NEW] 2.5D input construction: stack [s-1, s, s+1] as
        # 3 channels instead of repeating the same slice 3 times.
        # Boundary slices are handled by edge replication (pad
        # with the first/last slice) so every slice gets a triplet.
        # This gives the backbone genuine neighbouring-slice context
        # on each forward pass rather than redundant single-slice info.
        # -------------------------------------------------------
        padded = torch.cat([
            x_perm[:, :1],      # repeat first slice to cover s=0 boundary
            x_perm,
            x_perm[:, -1:],     # repeat last slice to cover s=n-1 boundary
        ], dim=1)               # (B, n_slices+2, C, h, w)

        flat = torch.cat([
            padded[:, 0:n_slices],      # s-1 channel
            padded[:, 1:n_slices+1],    # s   channel (centre)
            padded[:, 2:n_slices+2],    # s+1 channel
        ], dim=2).reshape(B * n_slices, 3, h_sz, w_sz)  # (B*n, 3, h, w)
        # -------------------------------------------------------

        # Resize in-plane to DINOv2 input size
        if h_sz != self.input_size or w_sz != self.input_size:
            flat = F.interpolate(
                flat,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False,
            )

        flat = flat.clamp(0, 1)
        flat = normalize_slices(flat)        # ImageNet stats — (B*n, 3, H, W)

        # Chunk DINOv2 forward passes to avoid OOM
        _chunk = 4
        if flat.shape[0] > _chunk:
            embeddings = torch.cat(
                [self.encode_slices(c) for c in flat.split(_chunk, dim=0)],
                dim=0,
            )
        else:
            embeddings = self.encode_slices(flat)                     # (B*n, 2*D)

        H_seq = rearrange(embeddings, '(b s) d -> b s d', b=B)        # (B, n, 2*D)

        v, a = self.pool(H_seq)    # (B, 2*D), (B, n_slices)
        return v, a


# ---------------------------------------------------------------
# Multi-Modal Fusion
# ---------------------------------------------------------------

class MultiModalFusion(nn.Module):
    """
    Fuses embeddings from multiple modalities (T1c + T2w) via a second
    stage of attention pooling with a shared learnable task query.

    embed_dim here should already be 2*backbone_embed_dim, passed in
    from AnyMC3D after the doubling is applied.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.pool = AttentionPool(embed_dim)

    def forward(self, modality_embeddings: list) -> torch.Tensor:
        """
        Args:
            modality_embeddings: list of [B, 2*D] tensors, one per modality
        Returns:
            [B, 2*D] fused patient-level embedding
        """
        stacked = torch.stack(modality_embeddings, dim=1)  # [B, M, 2*D]
        v, _    = self.pool(stacked)
        return v


# ---------------------------------------------------------------
# Full AnyMC3D v2 Model
# ---------------------------------------------------------------

class AnyMC3D(nn.Module):
    """
    AnyMC3D v2: Scalable 3D classifier adapted from 2D Foundation Models.

    Changes from v1:
      [NEW] 2.5D input: [s-1, s, s+1] stacked as 3 channels per slice
      [NEW] Per-slice embedding: concat([CLS], AttentionPool(patches)) -> 2*D
      [NEW] All downstream dims updated to reflect 2*D embedding size

    For meningioma molecular subtype classification (MG1-MG4):
      - Accepts T1c and/or T2w MRI volumes
      - Frozen DINOv2 backbone with per-modality LoRA adapters
      - Permutation-invariant attention pooling for slice fusion
      - Optional multi-modal fusion via a second attention pool
      - Linear classification head
    """

    def __init__(
        self,
        num_classes:   int   = 4,
        modalities:    list  = ['t1c'],
        backbone_name: str   = 'dinov2_vitl14',
        lora_rank:     int   = 8,
        lora_alpha:    int   = 16,
        input_size:    int   = 224,
        dropout:       float = 0.1,
        slice_axis:    int   = 3,
    ):
        super().__init__()

        self.num_classes   = num_classes
        self.modalities    = modalities
        self.backbone_name = backbone_name

        # Load DINOv2 backbone and freeze all weights
        print(f"Loading {backbone_name} from torch.hub...")
        backbone = torch.hub.load('facebookresearch/dinov2', backbone_name)
        for param in backbone.parameters():
            param.requires_grad = False

        # -------------------------------------------------------
        # [NEW] embed_dim is doubled because each slice embedding
        # is now concat([CLS], AttentionPool(patches)) -> 2*D.
        # All downstream modules (fusion, classifier) are built
        # with this doubled dimension.
        # -------------------------------------------------------
        backbone_embed_dim = backbone.embed_dim
        embed_dim          = backbone_embed_dim * 2   # 2*D
        self.embed_dim     = embed_dim
        print(f"Backbone embed_dim: {backbone_embed_dim} -> slice embed_dim: {embed_dim} (CLS+patch concat)")

        self.encoders = nn.ModuleDict()
        for modality in modalities:
            self.encoders[modality] = ModalityEncoder(
                backbone   = copy.deepcopy(backbone),
                embed_dim  = backbone_embed_dim,  # pass raw backbone dim; doubling happens inside
                lora_rank  = lora_rank,
                lora_alpha = lora_alpha,
                input_size = input_size,
                slice_axis = slice_axis,
            )

        # Multi-modal fusion (only when > 1 modality)
        # embed_dim is already 2*D here so MultiModalFusion gets the correct size
        if len(modalities) > 1:
            self.fusion = MultiModalFusion(embed_dim)

        # Classification head operates on the 2*D patient embedding
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nAnyMC3D v2 initialized:")
        print(f"  Modalities:       {modalities}")
        print(f"  Classes:          {num_classes}")
        print(f"  Backbone:         {backbone_name} (embed_dim={backbone_embed_dim}, frozen)")
        print(f"  Slice embed_dim:  {embed_dim} (CLS + attention-pooled patches)")
        print(f"  Trainable params: {n_trainable:,}")

    def forward(self, inputs: dict):
        """
        Args:
            inputs: dict mapping modality name -> [B, 1, H, W, S] tensor
                    Values should be pre-normalized to [0, 1]
        Returns:
            logits: [B, num_classes]
            aux:    dict with per-modality attention weights for heatmaps
        """
        aux                 = {}
        modality_embeddings = []

        for modality in self.modalities:
            if modality not in inputs:
                raise KeyError(
                    f"Modality '{modality}' not in inputs. Got: {list(inputs.keys())}"
                )
            v, attn = self.encoders[modality](inputs[modality])
            modality_embeddings.append(v)
            aux[f'attn_{modality}'] = attn

        if len(self.modalities) == 1:
            patient_embedding = modality_embeddings[0]
        else:
            patient_embedding = self.fusion(modality_embeddings)

        logits = self.classifier(patient_embedding)
        return logits, aux


# ---------------------------------------------------------------
# PyTorch Lightning Training Module
# ---------------------------------------------------------------

class AnyMC3DLightningModule(L.LightningModule):
    """
    PyTorch Lightning wrapper for AnyMC3D v2.

    Handles:
      - Focal loss for class imbalance (gamma=2, alpha=0.25 as per paper)
      - Separate LRs for LoRA layers vs head (as per paper)
      - Linear warmup + cosine annealing LR schedule
      - AUROC, balanced accuracy, F1 (macro) metrics
    """

    def __init__(
        self,
        num_classes:        int   = 4,
        modalities:         list  = ['t1c'],
        backbone_name:      str   = 'dinov2_vitl14',
        lora_rank:          int   = 8,
        lora_alpha:         int   = 16,
        input_size:         int   = 224,
        dropout:            float = 0.1,
        slice_axis:         int   = 3,
        # Optimizer
        lora_lr:            float = 5e-5,
        lora_weight_decay:  float = 1e-5,
        head_lr:            float = 5e-4,
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

        self.model = AnyMC3D(
            num_classes   = num_classes,
            modalities    = modalities,
            backbone_name = backbone_name,
            lora_rank     = lora_rank,
            lora_alpha    = lora_alpha,
            input_size    = input_size,
            dropout       = dropout,
            slice_axis    = slice_axis,
        )

        self.num_classes = num_classes
        self.modalities  = modalities

        if num_classes == 2:
            metric_kwargs = dict(task='binary')
        else:
            metric_kwargs = dict(task='multiclass', num_classes=num_classes)

        self.val_auroc    = AUROC(**metric_kwargs)
        self.val_acc      = Accuracy(**metric_kwargs)
        self.val_f1       = F1Score(average='macro', **metric_kwargs)
        self.val_bal_acc  = BalancedAccuracy(num_classes=num_classes)

        self.test_auroc   = AUROC(**metric_kwargs)
        self.test_acc     = Accuracy(**metric_kwargs)
        self.test_f1      = F1Score(average='macro', **metric_kwargs)
        self.test_bal_acc = BalancedAccuracy(num_classes=num_classes)

        self.test_preds  = []
        self.test_labels = []

    # ------------------------------------------------------------------
    # Focal Loss
    # ------------------------------------------------------------------

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)"""
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return (self.hparams.focal_alpha * (1 - pt) ** self.hparams.focal_gamma * ce).mean()

    # ------------------------------------------------------------------
    # Batch unpacking
    # ------------------------------------------------------------------

    def _unpack_batch(self, batch):
        """
        Supports two batch formats:
          Format 1 (multi-modal dict): ({'t1c': tensor, 't2w': tensor}, labels)
          Format 2 (single tensor):    (tensor, labels)
        """
        x, y, *_ = batch
        if isinstance(x, dict):
            return x, y
        return {self.modalities[0]: x}, y

    def _shared_step(self, batch):
        inputs, y = self._unpack_batch(batch)
        logits, _ = self.model(inputs)
        loss      = self.focal_loss(logits, y)
        probs     = F.softmax(logits, dim=1)
        return loss, probs, y

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True,
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
        self.log('val/loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        self.log('val/AUROC',             self.val_auroc.compute(),   prog_bar=True, sync_dist=True)
        self.log('val/accuracy',          self.val_acc.compute(),     prog_bar=True, sync_dist=True)
        self.log('val/F1_macro',          self.val_f1.compute(),      prog_bar=True, sync_dist=True)
        self.log('val/balanced_accuracy', self.val_bal_acc.compute(), prog_bar=True, sync_dist=True)
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

        self.log('test/AUROC',             auroc)
        self.log('test/accuracy',          acc)
        self.log('test/F1_macro',          f1)
        self.log('test/balanced_accuracy', balanced_acc)

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
        inputs, y = self._unpack_batch(batch)
        logits, aux = self.model(inputs)
        return y, logits

    # ------------------------------------------------------------------
    # Optimizer: separate LRs for LoRA vs head (as per paper)
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        lora_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ['classifier', 'pool.query', 'fusion']):
                head_params.append(param)
            else:
                lora_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': lora_params,
             'lr': self.hparams.lora_lr,
             'weight_decay': self.hparams.lora_weight_decay},
            {'params': head_params,
             'lr': self.hparams.head_lr,
             'weight_decay': self.hparams.head_weight_decay},
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
            CosineAnnealingLR(optimizer,
                              T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
                              )
        )

        if len(schedulers) == 1:
            scheduler = schedulers[0]
        else:
            scheduler = SequentialLR(optimizer,
                                     schedulers=schedulers,
                                     milestones=milestones)

        return {
            'optimizer':    optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }


# ---------------------------------------------------------------
# Quick sanity check — run with: python anymc3d_v2.py
# ---------------------------------------------------------------

if __name__ == '__main__':
    print("Running AnyMC3D v2 sanity check...\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    B, H, W, S = 2, 64, 64, 64

    dummy_inputs = {
        't1c': torch.randn(B, 1, H, W, S).clamp(0, 1).to(device),
        # 't2w': torch.randn(B, 1, H, W, S).clamp(0, 1).to(device),
    }

    # Use ViT-S for quick testing — swap to dinov2_vitl14 for real training
    model = AnyMC3D(
        num_classes   = 4,
        modalities    = ['t1c'],
        backbone_name = 'dinov2_vits14',
        lora_rank     = 8,
        lora_alpha    = 16,
        input_size    = 224,
    ).to(device)

    with torch.no_grad():
        logits, aux = model(dummy_inputs)

    print(f"Output logits shape:    {logits.shape}")           # [2, 4]
    print(f"Attn t1c shape:         {aux['attn_t1c'].shape}")  # [2, 64]
    print(f"Expected embed_dim:     {model.embed_dim}")        # 2 * backbone_dim
    print("\nSanity check passed!")
