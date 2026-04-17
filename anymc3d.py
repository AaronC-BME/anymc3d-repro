"""
AnyMC3D - Scalable 3D Medical Image Classifier adapted from 2D Foundation Models
Based on: "Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification"
         Liu et al., 2025 (arXiv:2512.12887)

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
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, H: torch.Tensor):
        """
        Args:
            H: [B, S, D] sequence of slice embeddings
        Returns:
            v: [B, D]    aggregated volume embedding
            a: [B, S]    attention weights (for heatmap visualization)
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
    """

    def __init__(
        self,
        backbone:    nn.Module,
        embed_dim:   int,
        lora_rank:   int = 8,
        lora_alpha:  int = 16,
        input_size:  int = 224,
        slice_axis:  int = 1,   # which spatial dim to slice along
                                # for (B,1,H,W,S): 1=H, 2=W, 3=S
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.input_size = input_size
        self.slice_axis = slice_axis

        lora_config = LoraConfig(
            r            = lora_rank,
            lora_alpha   = lora_alpha,
            target_modules = ["qkv", "proj", "patch_embed.proj"],
            lora_dropout = 0.0,
            bias         = "none",
        )
        self.adapted_backbone = get_peft_model(backbone, lora_config)

        # One query per modality encoder — this is the view query (q^(i)) from Fig 2b
        self.pool = AttentionPool(embed_dim)

    def encode_slices(self, x: torch.Tensor) -> torch.Tensor:
        out = self.adapted_backbone.forward_features(x)
        if isinstance(out, dict):
            return out['x_norm_clstoken']
        return out[:, 0]

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 1, H, W, S]  values in [0, 1]
               For PDCAD: [B, 1, 300, 300, 70]
               slice_axis=1 means we slice along H=300
        Returns:
            v:    [B, D]         modality embedding
            attn: [B, n_slices]  attention weights
        """
        B, C, H, W, S = x.shape

        # Move slice axis to position 1 so we can index along it
        # slice_axis refers to the spatial dims: 1=H, 2=W, 3=S
        # In the full tensor dims are: 0=B, 1=C, 2=H, 3=W, 4=S
        spatial_to_full = {1: 2, 2: 3, 3: 4}
        full_axis = spatial_to_full[self.slice_axis]

        # Move the slicing axis to dim 1 (after batch)
        # e.g. for H: (B,C,H,W,S) -> (B,H,C,W,S)
        perm        = [0, full_axis] + [i for i in range(1, 5) if i != full_axis]
        x_perm      = x.permute(*perm)           # (B, n_slices, C, h, w)
        n_slices    = x_perm.shape[1]
        h_sz        = x_perm.shape[3]
        w_sz        = x_perm.shape[4]

        # Flatten batch and slice dims
        flat = x_perm.reshape(B * n_slices, C, h_sz, w_sz)

        # Resize in-plane to DINOv2 input size
        if h_sz != self.input_size or w_sz != self.input_size:
            flat = F.interpolate(
                flat,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False,
            )

        flat = flat.clamp(0, 1)
        flat = flat.repeat(1, 3, 1, 1)       # (B*n, 3, 224, 224)
        flat = normalize_slices(flat)         # ImageNet stats

        # Chunk DINOv2 forward passes to avoid OOM at high resolutions
        # (e.g., 308x308 with 70 slices × batch_size 2 = 140 passes)
        _chunk = 4
        if flat.shape[0] > _chunk:
            embeddings = torch.cat(
                [self.encode_slices(c) for c in flat.split(_chunk, dim=0)],
                dim=0,
            )
        else:
            embeddings = self.encode_slices(flat)                   # (B*n, D)                      # (B*n, D)
        H_seq      = rearrange(embeddings, '(b s) d -> b s d', b=B) # (B, n, D)

        v, a = self.pool(H_seq)    # (B, D), (B, n_slices)
        return v, a


# ---------------------------------------------------------------
# Multi-Modal Fusion
# ---------------------------------------------------------------

class MultiModalFusion(nn.Module):
    """
    Fuses embeddings from multiple modalities (T1c + T2w) via a second
    stage of attention pooling with a shared learnable task query.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.pool = AttentionPool(embed_dim)

    def forward(self, modality_embeddings: list) -> torch.Tensor:
        """
        Args:
            modality_embeddings: list of [B, D] tensors, one per modality
        Returns:
            [B, D] fused patient-level embedding
        """
        stacked = torch.stack(modality_embeddings, dim=1)   # [B, M, D]
        v, _    = self.pool(stacked)
        return v


# ---------------------------------------------------------------
# Full AnyMC3D Model
# ---------------------------------------------------------------

class AnyMC3D(nn.Module):
    """
    AnyMC3D: Scalable 3D classifier adapted from 2D Foundation Models.

    Trainable parameters per modality (~1.2M):
        LoRA adapters + attention pool query + (fusion query) + classifier head
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
        """
        Args:
            num_classes:    Number of output classes (4 for MG1-MG4)
            modalities:     List of modality names, e.g. ['t1c'] or ['t1c', 't2w']
            backbone_name:  DINOv2 variant:
                              'dinov2_vitl14' — best performance (recommended)
                              'dinov2_vitb14' — faster, less memory
                              'dinov2_vits14' — fastest, for quick testing
            lora_rank:      LoRA rank r (paper uses 8)
            lora_alpha:     LoRA scaling alpha (paper uses 16)
            input_size:     Slice resize target (224 for standard DINOv2)
            dropout:        Dropout before classification head
        """
        super().__init__()

        self.num_classes   = num_classes
        self.modalities    = modalities
        self.backbone_name = backbone_name

        # Load DINOv2 backbone and freeze all weights
        print(f"Loading {backbone_name} from torch.hub...")
        backbone = torch.hub.load('facebookresearch/dinov2', backbone_name)
        for param in backbone.parameters():
            param.requires_grad = False

        embed_dim      = backbone.embed_dim
        self.embed_dim = embed_dim
        print(f"Backbone embed_dim: {embed_dim}")

        # In AnyMC3D.__init__, replace the encoders block with:
        self.encoders = nn.ModuleDict()
        for modality in modalities:
            self.encoders[modality] = ModalityEncoder(
                backbone   = copy.deepcopy(backbone),
                embed_dim  = embed_dim,
                lora_rank  = lora_rank,
                lora_alpha = lora_alpha,
                input_size = input_size,
                slice_axis = slice_axis,  # add this parameter to AnyMC3D.__init__ too
            )

        # Multi-modal fusion (only when > 1 modality)
        if len(modalities) > 1:
            self.fusion = MultiModalFusion(embed_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nAnyMC3D initialized:")
        print(f"  Modalities:       {modalities}")
        print(f"  Classes:          {num_classes}")
        print(f"  Backbone:         {backbone_name} (embed_dim={embed_dim}, frozen)")
        print(f"  Trainable params: {n_trainable:,}")

    def forward(self, inputs: dict):
        """
        Args:
            inputs: dict mapping modality name -> [B, 1, H, W, S] tensor
                    Values should be pre-normalized to [0, 1]
        Returns:
            logits: [B, num_classes]
            aux:    dict with per-modality per-plane attention weights for heatmaps
                    e.g. aux['attn_t1c'] = {'axial': [B,64], 'coronal': [B,64], 'sagittal': [B,64]}
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
            aux[f'attn_{modality}'] = attn  # dict of {plane: [B, n_slices]}

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
    PyTorch Lightning wrapper for AnyMC3D.

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
        lr_scheduler:   str   = "cosine",   
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
        )

        self.num_classes = num_classes
        self.modalities  = modalities

        # Metrics
        # With:
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

        # Store test predictions for downstream analysis
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
        self.val_auroc.update(probs[:, 1], y)  # <-- fix: pass prob of class 1 only
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
        self.test_auroc.update(probs[:, 1], y)  # <-- fix this line
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

        # cosine (with optional warmup)
        schedulers = []
        milestones = []

        if self.hparams.warmup_epochs > 0:
            schedulers.append(
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                        total_iters=self.hparams.warmup_epochs)
            )
            milestones.append(self.hparams.warmup_epochs)

        schedulers.append(
            CosineAnnealingLR(optimizer,
                            T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
                            eta_min=1e-6)
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
# Quick sanity check — run with: python anymc3d.py
# ---------------------------------------------------------------

if __name__ == '__main__':
    print("Running AnyMC3D sanity check...\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    B, H, W, S = 2, 64, 64, 64

    dummy_inputs = {
        't1c': torch.randn(B, 1, H, W, S).clamp(0, 1).to(device),
        # 't2w': torch.randn(B, 1, H, W, S).clamp(0, 1).to(device),
    }

    # Use ViT-S for quick testing — swap to dinov2_vitl14 for real training
    model = AnyMC3D(
        num_classes   = 2,
        modalities    = ['t1c'],
        backbone_name = 'dinov2_vits14',
        lora_rank     = 8,
        lora_alpha    = 16,
        input_size    = 224,
    ).to(device)

    with torch.no_grad():
        logits, aux = model(dummy_inputs)

    print(f"Output logits shape: {logits.shape}")          # [2, 4]
    for modality in ['t1c', 't2w']:
        for plane, weights in aux[f'attn_{modality}'].items():
            print(f"Attention [{modality}][{plane}]: {weights.shape}")  # [2, 64]
    print("\nSanity check passed!")
