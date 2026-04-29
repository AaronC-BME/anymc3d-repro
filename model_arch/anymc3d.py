"""
AnyMC3D Backbone — AnyMC3D with two learnable vision blocks on top of DINOv2
Based on: "Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification"
         Liu et al., 2025 (arXiv:2512.12887)

         + "DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level
            Vision-Language Alignment" Jose et al., 2024 (arXiv:2412.16334)

Changes from v1 (anymc3d.py):
  [v2] Two learnable transformer blocks (VisionBlocks) are added on top of the
       frozen DINOv2 backbone, following dino.txt Section 3.1.

Changes from v2 (anymc3d_backbone.py):
  [v3 - NEW] use_patch_concat flag:
       When True, the mean-pooled patch tokens are concatenated with the
       adapted CLS token to form a 2D-dimensional slice embedding instead
       of D. This doubles the slice-level feature richness at the cost of
       doubling the AttentionPool, fusion, and classifier head dimensions.
       AttentionPool, MultiModalFusion, and the linear head all automatically
       use slice_embed_dim = embed_dim * 2 when this is enabled.

  [v3 - NEW] use_25d flag:
       When True, each slice is represented as a triplet of consecutive
       slices [s-1, s, s+1] stacked along the channel axis (2.5D input).
       Boundary slices are padded by repeating the first/last slice.
       This replaces the single-slice-replicated-to-RGB approach.
       When False (default), the original single-slice replication is used.

Everything else is identical to v2:
  - Single-slice input (one slice replicated to 3 channels) [when use_25d=False]
  - CLS token only as slice embedding [when use_patch_concat=False]
  - Same attention pooling, multi-modal fusion, and classifier head

Adapted for Meningioma Molecular Subtype Classification (MG1-MG4)
using T1-contrast and T2-weighted MRI data.

Usage:
    # Default (v2 behaviour)
    model = AnyMC3D(num_classes=4, modalities=['t1c'])

    # With both new features enabled
    model = AnyMC3D(
        num_classes=4,
        modalities=['t1c'],
        use_patch_concat=True,
        use_25d=True,
    )
"""

import copy
import math

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
# Learnable Vision Blocks
# ---------------------------------------------------------------

class VisionBlock(nn.Module):
    """
    A single standard transformer block: multi-head self-attention + MLP.
    Preserves input dimensionality D throughout — identical to the blocks
    inside DINOv2 but trained from scratch rather than frozen.

    Used to build the two-block adapter ψ from dino.txt Section 3.1.
    Processes the full token sequence [CLS, patch1, ..., patchN] output
    by the frozen backbone, producing an adapted sequence of the same shape.
    """

    def __init__(
        self,
        embed_dim:   int,
        num_heads:   int,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,   # expects [B, seq, D]
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1+N, D]  full token sequence (CLS + patch tokens)
        Returns:
            x: [B, 1+N, D]  adapted token sequence, same shape
        """
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionBlocks(nn.Module):
    """
    Two stacked VisionBlock layers — the adapter ψ from dino.txt.

    Takes the full token sequence [CLS, f1, ..., fN] from the frozen DINOv2
    backbone and produces an adapted sequence [c', f1', ..., fN'] of the same
    shape.
        ViT-S: D=384,  heads=6
        ViT-B: D=768,  heads=12
        ViT-L: D=1024, heads=16
    """

    _HEAD_MAP = {384: 6, 768: 12, 1024: 16}

    def __init__(
        self,
        embed_dim:    int,
        num_blocks:   int   = 2,
        mlp_ratio:    float = 4.0,
        dropout:      float = 0.0,
    ):
        super().__init__()
        num_heads = self._HEAD_MAP.get(embed_dim, max(1, embed_dim // 64))
        self.blocks = nn.Sequential(*[
            VisionBlock(
                embed_dim  = embed_dim,
                num_heads  = num_heads,
                mlp_ratio  = mlp_ratio,
                dropout    = dropout,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, 1+N, D]  CLS token prepended to N patch tokens
        Returns:
                    [B, 1+N, D]  adapted tokens, same shape
        """
        return self.blocks(tokens)


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

    New flags vs v2:
      use_patch_concat (bool):
        If True, the slice embedding is [CLS ; mean(patch tokens)], giving a
        2D-dimensional vector. AttentionPool is sized accordingly.
        If False (default), only the adapted CLS token is used (v2 behaviour).

      use_25d (bool):
        If True, each slice is encoded with its two neighbours as a 3-channel
        2.5D input [s-1, s, s+1]. Boundary slices are padded by repeating.
        If False (default), a single slice is replicated to 3 channels (v2).
    """

    def __init__(
        self,
        backbone:           nn.Module,
        embed_dim:          int,
        lora_rank:          int   = 8,
        lora_alpha:         int   = 16,
        input_size:         int   = 224,
        slice_axis:         int   = 3,
        vision_blocks:      int   = 2,
        mlp_ratio:          float = 4.0,
        block_dropout:      float = 0.0,
        # AnyMC3D feature
        use_slice_attn_pool: bool = True,   # True: AttentionPool over slices

        # [Optional features]
        use_patch_concat:   bool  = False,
        use_25d:            bool  = False,
        use_patch_attn_pool: bool = False,   # True: AttentionPool over patches
                                              # False: mean pooling (default)

    ):
        super().__init__()
        self.embed_dim            = embed_dim
        self.input_size           = input_size
        self.slice_axis           = slice_axis
        self.use_slice_attn_pool  = use_slice_attn_pool
        self.use_patch_concat     = use_patch_concat
        self.use_25d              = use_25d
        self.use_patch_attn_pool  = use_patch_attn_pool

        # Slice embedding dimensionality: doubled when patch tokens are concat'd
        self.slice_embed_dim = embed_dim * 2 if use_patch_concat else embed_dim

        lora_config = LoraConfig(
            r              = lora_rank,
            lora_alpha     = lora_alpha,
            target_modules = ["qkv", "proj", "patch_embed.proj"],
            lora_dropout   = 0.0,
            bias           = "none",
        )
        self.adapted_backbone = get_peft_model(backbone, lora_config)

        if vision_blocks > 0:
            self.vision_blocks = VisionBlocks(
                embed_dim  = embed_dim,
                num_blocks = vision_blocks,
                mlp_ratio  = mlp_ratio,
                dropout    = block_dropout,
            )

        if use_slice_attn_pool:
            self.pool = AttentionPool(self.slice_embed_dim)

        # Optional learned patch pooling (only instantiated when both flags are active)
        if use_patch_concat and use_patch_attn_pool:
            self.patch_pool = AttentionPool(embed_dim)

    def encode_slices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract per-slice embeddings from the adapted backbone.

        v2:  returns adapted CLS token c'                    -> [B, D]
        [use_patch_concat=True, use_patch_attn_pool=False]:
             returns [c' ; mean(f1',...,fN')]                -> [B, 2D]
        [use_patch_concat=True, use_patch_attn_pool=True]:
             returns [c' ; AttentionPool(f1',...,fN')]       -> [B, 2D]
        """
        out = self.adapted_backbone.forward_features(x)

        # Reconstruct full token sequence [CLS, patch1, ..., patchN]
        if isinstance(out, dict):
            cls_token    = out['x_norm_clstoken']       # [B, D]
            patch_tokens = out['x_norm_patchtokens']    # [B, N, D]
            tokens = torch.cat(
                [cls_token.unsqueeze(1), patch_tokens], dim=1
            )                                           # [B, 1+N, D]
        else:
            tokens = out                                # [B, 1+N, D]

        # Pass full token sequence through two learnable vision blocks
        if hasattr(self, 'vision_blocks'):
            tokens = self.vision_blocks(tokens)             # [B, 1+N, D]

        # Extract adapted CLS token c' from position 0
        cls_adapted = tokens[:, 0]                      # [B, D]

        if not self.use_patch_concat:
            return cls_adapted                          # [B, D]

        # Pool adapted patch tokens, then concatenate with CLS -> [B, 2D]
        patch_tokens_adapted = tokens[:, 1:]            # [B, N, D]
        if self.use_patch_attn_pool:
            patch_repr, _ = self.patch_pool(patch_tokens_adapted)  # [B, D]
        else:
            patch_repr = patch_tokens_adapted.mean(dim=1)          # [B, D]

        return torch.cat([cls_adapted, patch_repr], dim=-1)        # [B, 2D]

    def _build_25d_input(self, x_perm: torch.Tensor) -> torch.Tensor:
        """
        [NEW v3] Build 2.5D input by stacking each slice with its neighbours.

        Args:
            x_perm: [B, n_slices, C, H, W]  C=1 for single-channel MRI
        Returns:
            flat:   [B*n_slices, 3, H, W]   3-channel triplet per slice
        """
        B, S, C, H, W = x_perm.shape

        # Pad at boundaries by repeating the first and last slice
        x_padded = torch.cat(
            [x_perm[:, :1], x_perm, x_perm[:, -1:]], dim=1
        )                                               # [B, S+2, C, H, W]

        # Stack triplets [s-1, s, s+1] along channel axis
        prev_slices = x_padded[:, :-2]                  # [B, S, C, H, W]
        curr_slices = x_padded[:, 1:-1]                 # [B, S, C, H, W]
        next_slices = x_padded[:, 2:]                   # [B, S, C, H, W]

        triplets = torch.cat(
            [prev_slices, curr_slices, next_slices], dim=2
        )                                               # [B, S, 3C, H, W]
        # Since C=1 for medical images, 3C=3 — exactly the RGB format expected.

        return triplets.reshape(B * S, 3 * C, H, W)    # [B*S, 3, H, W]

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 1, H, W, S]  values in [0, 1]
        Returns:
            v:    [B, slice_embed_dim]  modality embedding
            attn: [B, n_slices]         attention weights
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

        if self.use_25d:
            # [NEW v3] Stack [s-1, s, s+1] as a 3-channel 2.5D input.
            # _build_25d_input handles the boundary padding and reshaping.
            flat = self._build_25d_input(x_perm)        # (B*n, 3, h, w)
        else:
            # Original v2: flatten batch+slice, then replicate single channel
            flat = x_perm.reshape(B * n_slices, C, h_sz, w_sz)
            flat = flat.repeat(1, 3, 1, 1)              # (B*n, 3, H, W)

        # Resize in-plane to DINOv2 input size
        if h_sz != self.input_size or w_sz != self.input_size:
            flat = F.interpolate(
                flat,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False,
            )

        flat = flat.clamp(0, 1)
        flat = normalize_slices(flat)         # ImageNet stats

        # Chunk DINOv2 forward passes to avoid OOM
        _chunk = 4
        if flat.shape[0] > _chunk:
            embeddings = torch.cat(
                [self.encode_slices(c) for c in flat.split(_chunk, dim=0)],
                dim=0,
            )
        else:
            embeddings = self.encode_slices(flat)       # (B*n, slice_embed_dim)

        H_seq = rearrange(embeddings, '(b s) d -> b s d', b=B)  # (B, S, D)

        if self.use_slice_attn_pool:
            v, a = self.pool(H_seq)                          # paper method
        else:
            v = H_seq.mean(dim=1)                            # mean pooling ablation
            a = torch.ones(B, H_seq.shape[1],
                        device=H_seq.device) / H_seq.shape[1]  # uniform weights
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
        stacked = torch.stack(modality_embeddings, dim=1)  # [B, M, D]
        v, _    = self.pool(stacked)
        return v


# ---------------------------------------------------------------
# Full AnyMC3D Backbone Model
# ---------------------------------------------------------------

class AnyMC3D(nn.Module):
    """
    AnyMC3D Backbone with optional CLS+patch concat and 2.5D slice input.

    Architecture:
      Frozen DINOv2 + LoRA  ->  VisionBlocks (2x, trainable)
      ->  [CLS ; mean(patches)] or CLS only
      ->  AttentionPool (slice fusion)
      ->  [MultiModalFusion]
      ->  Linear head

    New flags:
      use_patch_concat: if True, slice embedding = [CLS ; mean_patch], dim=2D
      use_25d:          if True, input = [s-1, s, s+1] triplet (2.5D)

    Trainable parameters:
      - LoRA adapters inside DINOv2      (~0.5M for ViT-B)
      - VisionBlocks (2 transformer blocks) (~14M for ViT-B, D=768)
      - AttentionPool query              (~768 or ~1536 params)
      - Fusion query (if multi-modal)    (~768 or ~1536 params)
      - Linear classifier head
    """

    def __init__(
        self,
        num_classes:        int   = 4,
        modalities:         list  = ['t1c'],
        backbone_name:      str   = 'dinov2_vitb14',
        lora_rank:          int   = 8,
        lora_alpha:         int   = 16,
        input_size:         int   = 224,
        cls_head_dropout:   float = 0.1,
        slice_axis:         int   = 3,
        vision_blocks:      int   = 2,
        mlp_ratio:          float = 4.0,
        block_dropout:      float = 0.0,
        # [NEW v3]
        use_slice_attn_pool: bool  = True,
        use_patch_concat:   bool  = False,
        use_25d:            bool  = False,
        use_patch_attn_pool: bool = False,
    ):
        super().__init__()

        self.num_classes         = num_classes
        self.modalities          = modalities
        self.backbone_name       = backbone_name
        self.use_slice_attn_pool = use_slice_attn_pool
        self.use_patch_concat    = use_patch_concat
        self.use_25d             = use_25d
        self.use_patch_attn_pool = use_patch_attn_pool

        print(f"Loading {backbone_name} from torch.hub...")
        backbone = torch.hub.load('facebookresearch/dinov2', backbone_name)
        for param in backbone.parameters():
            param.requires_grad = False

        embed_dim      = backbone.embed_dim
        self.embed_dim = embed_dim
        print(f"Backbone embed_dim: {embed_dim}")

        self.encoders = nn.ModuleDict()
        for modality in modalities:
            self.encoders[modality] = ModalityEncoder(
                backbone             = copy.deepcopy(backbone),
                embed_dim            = embed_dim,
                lora_rank            = lora_rank,
                lora_alpha           = lora_alpha,
                input_size           = input_size,
                slice_axis           = slice_axis,
                vision_blocks        = vision_blocks,
                mlp_ratio            = mlp_ratio,
                block_dropout        = block_dropout,
                use_slice_attn_pool  = use_slice_attn_pool,
                use_patch_concat     = use_patch_concat,
                use_25d              = use_25d,
                use_patch_attn_pool  = use_patch_attn_pool,
            )

        # Slice embedding dimensionality — 2D when patch concat is active
        slice_embed_dim      = embed_dim * 2 if use_patch_concat else embed_dim
        self.slice_embed_dim = slice_embed_dim

        # Multi-modal fusion uses slice_embed_dim throughout
        if len(modalities) > 1:
            self.fusion = MultiModalFusion(slice_embed_dim)

        # Classification head — input is slice_embed_dim
        self.classifier = nn.Sequential(
            nn.Dropout(cls_head_dropout),
            nn.Linear(slice_embed_dim, num_classes),
        )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if vision_blocks > 0:
            n_vision_block_params = sum(
                p.numel()
                for enc in self.encoders.values()
                for p in enc.vision_blocks.parameters()
                if p.requires_grad
            )
        print(f"\nAnyMC3D Backbone initialized:")
        print(f"  Modalities:             {modalities}")
        print(f"  Classes:                {num_classes}")
        print(f"  Backbone:               {backbone_name} (embed_dim={embed_dim}, frozen)")
        print(f"  Vision blocks:          {vision_blocks} x transformer block (trainable)")
        print(f"  Vision block params:    {n_vision_block_params:,} per modality" if vision_blocks > 0 else "0 (no vision blocks)")
        print(f"  Slice embed dim:        {slice_embed_dim} (patch_concat={use_patch_concat})")
        print(f"  Slice pooling:          {'AttentionPool' if use_slice_attn_pool else 'mean'} (use_slice_attn_pool={use_slice_attn_pool})")
        print(f"  Patch pooling:          {'AttentionPool' if use_patch_attn_pool else 'mean'} (use_patch_attn_pool={use_patch_attn_pool})")
        print(f"  2.5D input:             {use_25d}")
        print(f"  Total trainable params: {n_trainable:,}")

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
    PyTorch Lightning wrapper for AnyMC3D Backbone.

    Handles:
      - Focal loss for class imbalance (gamma=2, alpha=0.25 as per paper)
      - Three-way parameter grouping for optimizer:
          (1) LoRA adapter params       -> lora_lr
          (2) Vision block params       -> vision_lr
          (3) Head params               -> head_lr
      - Linear warmup + cosine annealing LR schedule
      - AUROC, balanced accuracy, F1 (macro) metrics
    """

    def __init__(
        self,
        num_classes:          int   = 4,
        modalities:           list  = ['t1c'],
        task:                 str   = "multiclass",  
        backbone_name:        str   = 'dinov2_vitl14',
        lora_rank:            int   = 8,
        lora_alpha:           int   = 16,
        input_size:           int   = 224,
        cls_head_dropout:     float = 0.1,
        slice_axis:           int   = 3,
        vision_blocks:        int   = 2,
        mlp_ratio:            float = 4.0,
        block_dropout:        float = 0.0,
        # [NEW v3]
        use_slice_attn_pool:  bool  = True,
        use_patch_concat:     bool  = False,
        use_25d:              bool  = False,
        use_patch_attn_pool:  bool  = False,
        # Optimizer
        lora_lr:              float = 5e-5,
        lora_weight_decay:    float = 1e-5,
        vision_lr:            float = 1e-4,
        vision_weight_decay:  float = 1e-5,
        head_lr:              float = 5e-4,
        head_weight_decay:    float = 1e-4,
        warmup_epochs:        int   = 10,
        max_epochs:           int   = 150,
        lr_scheduler:         str   = "cosine",
        # Focal loss
        focal_gamma:          float = 2.0,
        focal_alpha:          float = 0.25,
        # Training hyperparameters
        seed:                 int   = 0,
        precision:            str   = "16-mixed",
        early_stopping_patience: int = 50,
        save_top_k:           int   = 3,
        log_every_n_steps:    int   = 10,
        devices:              int = 1,
        strategy:            str   = "auto",
        num_nodes:           int   = 1,
        sync_batchnorm:      bool  = False,
        #WandB logging
        project:              str   = "AnyMC3D",
        run_name:             str   = "AnyMC3D_Run",
    ):
        super().__init__()
        self.save_hyperparameters()

        assert task in ('multiclass', 'multilabel'), f"Unknown task: {task}"
        self.task = task

        self.model = AnyMC3D(
            num_classes          = num_classes,
            modalities           = modalities,
            backbone_name        = backbone_name,
            lora_rank            = lora_rank,
            lora_alpha           = lora_alpha,
            input_size           = input_size,
            cls_head_dropout     = cls_head_dropout,
            slice_axis           = slice_axis,
            vision_blocks        = vision_blocks,
            mlp_ratio            = mlp_ratio,
            block_dropout        = block_dropout,
            use_slice_attn_pool  = use_slice_attn_pool,
            use_patch_concat     = use_patch_concat,
            use_25d              = use_25d,
            use_patch_attn_pool  = use_patch_attn_pool,
        )

        self.num_classes = num_classes
        self.modalities  = modalities

        if task == 'multilabel':
            metric_kwargs = dict(task='multilabel', num_labels=num_classes)
        elif num_classes == 2:
            metric_kwargs = dict(task='binary')
        else:
            metric_kwargs = dict(task='multiclass', num_classes=num_classes)

        # ── Training metrics (mirror val) ─────────────────────────────────────
        self.train_auroc = AUROC(average='macro', **metric_kwargs) if task == 'multilabel' else AUROC(**metric_kwargs)
        self.train_acc   = Accuracy(average='macro', **metric_kwargs) if task == 'multilabel' else Accuracy(**metric_kwargs)
        self.train_f1    = F1Score(average='macro', **metric_kwargs)

        if task != 'multilabel':
            self.train_bal_acc = BalancedAccuracy(num_classes=num_classes)

        self.val_auroc    = AUROC(average='macro', **metric_kwargs) if task == 'multilabel' else AUROC(**metric_kwargs)
        self.val_acc      = Accuracy(average='macro', **metric_kwargs) if task == 'multilabel' else Accuracy(**metric_kwargs)
        self.val_f1       = F1Score(average='macro', **metric_kwargs)

        # BalancedAccuracy is a multiclass concept — skip it for multilabel
        if task != 'multilabel':
            self.val_bal_acc = BalancedAccuracy(num_classes=num_classes)

        self.test_auroc   = AUROC(average='macro', **metric_kwargs) if task == 'multilabel' else AUROC(**metric_kwargs)
        self.test_acc     = Accuracy(average='macro', **metric_kwargs) if task == 'multilabel' else Accuracy(**metric_kwargs)
        self.test_f1      = F1Score(average='macro', **metric_kwargs)
        if task != 'multilabel':
            self.test_bal_acc = BalancedAccuracy(num_classes=num_classes)

        self.test_preds  = []
        self.test_labels = []

    # ------------------------------------------------------------------
    # Focal Loss
    # ------------------------------------------------------------------

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Focal loss with two modes:
        - multiclass: softmax-based, targets are class indices [B]
        - multilabel: sigmoid-based, targets are multi-hot floats [B, K]
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
        """
        if self.task == 'multilabel':
            # targets must be float for BCE
            targets = targets.float()
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            pt = torch.exp(-bce)  # prob of correct class per element
            focal = self.hparams.focal_alpha * (1 - pt) ** self.hparams.focal_gamma * bce
            return focal.mean()
        else:
            ce = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce)
            return (self.hparams.focal_alpha * (1 - pt) ** self.hparams.focal_gamma * ce).mean()

    # ------------------------------------------------------------------
    # Batch unpacking
    # ------------------------------------------------------------------

    def _unpack_batch(self, batch):
        x, y, *_ = batch
        if isinstance(x, dict):
            return x, y
        return {self.modalities[0]: x}, y

    def _shared_step(self, batch):
        inputs, y = self._unpack_batch(batch)
        logits, _ = self.model(inputs)
        loss      = self.focal_loss(logits, y)
        if self.task == 'multilabel':
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)
        return loss, probs, y

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch)

        if self.task == 'multilabel':
            y_int = y.int()
            self.train_auroc.update(probs, y_int)
            self.train_acc.update(probs, y_int)
            self.train_f1.update(probs, y_int)
        else:
            preds = probs.argmax(dim=1)
            auroc_input = probs[:, 1] if self.num_classes == 2 else probs
            self.train_auroc.update(auroc_input, y)
            self.train_acc.update(preds, y)
            self.train_f1.update(preds, y)
            self.train_bal_acc.update(preds, y)

        self.log('train/loss', loss, on_step=True, on_epoch=True,
                prog_bar=True, sync_dist=True)
        return loss
    
    def on_train_epoch_end(self):
        self.log('train/AUROC',    self.train_auroc.compute(), prog_bar=False, sync_dist=True)
        self.log('train/accuracy', self.train_acc.compute(),   prog_bar=False, sync_dist=True)
        self.log('train/F1_macro', self.train_f1.compute(),    prog_bar=False, sync_dist=True)
        self.train_auroc.reset(); self.train_acc.reset(); self.train_f1.reset()

        if self.task != 'multilabel':
            self.log('train/balanced_accuracy', self.train_bal_acc.compute(),
                    prog_bar=False, sync_dist=True)
            self.train_bal_acc.reset()

    def validation_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch)

        if self.task == 'multilabel':
            # torchmetrics multilabel expects int targets [B, K] and float probs [B, K]
            y_int = y.int()
            self.val_auroc.update(probs, y_int)
            self.val_acc.update(probs, y_int)
            self.val_f1.update(probs, y_int)
        else:
            preds = probs.argmax(dim=1)
            auroc_input = probs[:, 1] if self.num_classes == 2 else probs
            self.val_auroc.update(auroc_input, y)
            self.val_acc.update(preds, y)
            self.val_f1.update(preds, y)
            self.val_bal_acc.update(preds, y)

        self.log('val/loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self):
        self.log('val/AUROC',    self.val_auroc.compute(), prog_bar=True, sync_dist=True)
        self.log('val/accuracy', self.val_acc.compute(),   prog_bar=True, sync_dist=True)
        self.log('val/F1_macro', self.val_f1.compute(),    prog_bar=True, sync_dist=True)
        self.val_auroc.reset(); self.val_acc.reset(); self.val_f1.reset()

        if self.task != 'multilabel':
            self.log('val/balanced_accuracy', self.val_bal_acc.compute(),
                    prog_bar=True, sync_dist=True)
            self.val_bal_acc.reset()

    def test_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch)

        if self.task == 'multilabel':
            y_int = y.int()
            self.test_auroc.update(probs, y_int)
            self.test_acc.update(probs, y_int)
            self.test_f1.update(probs, y_int)
        else:
            preds = probs.argmax(dim=1)
            auroc_input = probs[:, 1] if self.num_classes == 2 else probs
            self.test_auroc.update(auroc_input, y)
            self.test_acc.update(preds, y)
            self.test_f1.update(preds, y)
            self.test_bal_acc.update(preds, y)

        self.test_preds.append(probs.cpu())
        self.test_labels.append(y.cpu())

    def on_test_epoch_end(self):
        auroc = self.test_auroc.compute()
        acc   = self.test_acc.compute()
        f1    = self.test_f1.compute()

        self.log('test/AUROC',    auroc)
        self.log('test/accuracy', acc)
        self.log('test/F1_macro', f1)

        print(f"\n{'='*50}")
        print(f"Test Results:")
        print(f"  AUROC (macro):     {auroc:.4f}")
        print(f"  Accuracy (macro):  {acc:.4f}")
        print(f"  F1 (macro):        {f1:.4f}")

        if self.task != 'multilabel':
            balanced_acc = self.test_bal_acc.compute()
            self.log('test/balanced_accuracy', balanced_acc)
            print(f"  Balanced Accuracy: {balanced_acc:.4f}")
            self.test_bal_acc.reset()

        print(f"{'='*50}\n")
        self.test_auroc.reset(); self.test_acc.reset(); self.test_f1.reset()

    def predict_step(self, batch, batch_idx):
        inputs, y = self._unpack_batch(batch)
        logits, aux = self.model(inputs)
        return y, logits

    # ------------------------------------------------------------------
    # Optimizer: three-way LR grouping
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        lora_params   = []
        vision_params = []
        head_params   = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in ['classifier', 'pool.query', 'fusion']):
                head_params.append(param)
            elif 'vision_blocks' in name:
                vision_params.append(param)
            else:
                lora_params.append(param)

        optimizer = torch.optim.AdamW([
            {'params': lora_params,
             'lr': self.hparams.lora_lr,
             'weight_decay': self.hparams.lora_weight_decay},
            {'params': vision_params,
             'lr': self.hparams.vision_lr,
             'weight_decay': self.hparams.vision_weight_decay},
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
# Quick sanity check — run with: python anymc3d_backbone.py
# ---------------------------------------------------------------

if __name__ == '__main__':
    print("Running AnyMC3D Backbone sanity check...\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    B, H, W, S = 2, 64, 64, 64

    dummy_inputs = {
        't1c': torch.randn(B, 1, H, W, S).clamp(0, 1).to(device),
    }

    print("=" * 60)
    print("Case 1: default (v2 behaviour — CLS only, replicated slice)")
    model_v2 = AnyMC3D(
        num_classes       = 4,
        modalities        = ['t1c'],
        backbone_name     = 'dinov2_vits14',
        lora_rank         = 8,
        lora_alpha        = 16,
        input_size        = 224,
        vision_blocks     = 2,
        use_patch_concat  = False,
        use_25d           = False,
    ).to(device)
    with torch.no_grad():
        logits, aux = model_v2(dummy_inputs)
    print(f"  logits shape:  {logits.shape}")           # [2, 4]
    print(f"  attn shape:    {aux['attn_t1c'].shape}")  # [2, 64]
    print(f"  embed_dim:     {model_v2.embed_dim}")
    print(f"  slice_embed:   {model_v2.slice_embed_dim}")

    print()
    print("=" * 60)
    print("Case 2: use_patch_concat=True, use_25d=True, mean pooling (default)")
    model_v3 = AnyMC3D(
        num_classes          = 4,
        modalities           = ['t1c'],
        backbone_name        = 'dinov2_vits14',
        lora_rank            = 8,
        lora_alpha           = 16,
        input_size           = 224,
        vision_blocks        = 2,
        use_patch_concat     = True,
        use_25d              = True,
        use_patch_attn_pool  = False,
    ).to(device)
    with torch.no_grad():
        logits, aux = model_v3(dummy_inputs)
    print(f"  logits shape:  {logits.shape}")              # [2, 4]
    print(f"  attn shape:    {aux['attn_t1c'].shape}")     # [2, 64]
    print(f"  embed_dim:     {model_v3.embed_dim}")        # D (e.g. 384)
    print(f"  slice_embed:   {model_v3.slice_embed_dim}")  # 2D (e.g. 768)

    print()
    print("=" * 60)
    print("Case 3: use_patch_concat=True, use_25d=True, AttentionPool over patches")
    model_v3_attn = AnyMC3D(
        num_classes          = 4,
        modalities           = ['t1c'],
        backbone_name        = 'dinov2_vits14',
        lora_rank            = 8,
        lora_alpha           = 16,
        input_size           = 224,
        vision_blocks        = 2,
        use_patch_concat     = True,
        use_25d              = True,
        use_patch_attn_pool  = True,
    ).to(device)
    with torch.no_grad():
        logits, aux = model_v3_attn(dummy_inputs)
    print(f"  logits shape:  {logits.shape}")              # [2, 4]
    print(f"  attn shape:    {aux['attn_t1c'].shape}")     # [2, 64]
    print(f"  embed_dim:     {model_v3_attn.embed_dim}")
    print(f"  slice_embed:   {model_v3_attn.slice_embed_dim}")  # 2D (e.g. 768)

    print("\nSanity check passed!")