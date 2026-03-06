"""
RSNA-Kaggle 2.5D Solution for Meningioma Molecular Subtype Classification
Based on the 1st-place solution from RSNA Abdominal Trauma Detection 2023

Architecture:
    1. Resample volume to fixed 96 slices along depth axis (trilinear interpolation)
    2. Group into 32 triplets of 3 adjacent slices (2.5D)
    3. 2D CNN backbone encodes each triplet -> 32 feature vectors (shared weights)
    4. GRU processes feature vectors sequentially (order matters)
    5. Linear classifier on mean-pooled GRU output

Usage:
    python rsna_kaggle_model.py   # sanity check
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import lightning as L
from torchmetrics import AUROC, Accuracy, F1Score
from balanced_accuracy import BalancedAccuracy
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


# ---------------------------------------------------------------
# ImageNet normalization
# ---------------------------------------------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(x.device).view(1, 3, 1, 1)
    std  = IMAGENET_STD.to(x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------------------------------------------------
# Volume preprocessing
# ---------------------------------------------------------------

def resample_volume(x: torch.Tensor, target_slices: int = 96) -> torch.Tensor:
    """
    Resample a 3D volume to a fixed number of slices along the depth axis
    using trilinear interpolation.

    Args:
        x:             [B, C, H, W, S] volume, values in [0, 1]
        target_slices: target slice count (default 96 = 32 triplets × 3)
    Returns:
        [B, C, H, W, target_slices]
    """
    B, C, H, W, S = x.shape
    if S == target_slices:
        return x
    # F.interpolate 3D expects [B, C, D, H, W]
    x = x.permute(0, 1, 4, 2, 3)                           # [B, C, S, H, W]
    x = F.interpolate(x, size=(target_slices, H, W),
                      mode='trilinear', align_corners=False)
    x = x.permute(0, 1, 3, 4, 2)                           # [B, C, H, W, target_slices]
    return x


def volume_to_triplets(x: torch.Tensor,
                       n_triplets: int = 32,
                       img_size: int = 224) -> torch.Tensor:
    """
    Convert a resampled volume into triplets of 3 adjacent slices.
    Each triplet is stacked as a 3-channel (RGB-like) image — the 2.5D trick.

    Args:
        x:          [B, 1, H, W, 96] resampled single-channel volume
        n_triplets: number of triplets (96 / 3 = 32)
        img_size:   spatial size to resize each triplet to
    Returns:
        [B, 32, 3, img_size, img_size]
    """
    B, C, H, W, S = x.shape
    assert S == n_triplets * 3, \
        f"Expected {n_triplets * 3} slices for {n_triplets} triplets, got {S}"

    # [B, H, W, 96] -> [B, 96, H, W] -> [B, 32, 3, H, W]
    x = x.squeeze(1)                                        # [B, H, W, 96]
    x = x.permute(0, 3, 1, 2)                              # [B, 96, H, W]
    x = x.reshape(B, n_triplets, 3, H, W)                  # [B, 32, 3, H, W]

    # Resize spatial dims if needed
    if H != img_size or W != img_size:
        x_flat = x.reshape(B * n_triplets, 3, H, W)
        x_flat = F.interpolate(x_flat, size=(img_size, img_size),
                               mode='bilinear', align_corners=False)
        x = x_flat.reshape(B, n_triplets, 3, img_size, img_size)

    # Clamp and apply ImageNet normalization
    x_flat = x.reshape(B * n_triplets, 3, img_size, img_size).clamp(0, 1)
    x_flat = normalize(x_flat)
    return x_flat.reshape(B, n_triplets, 3, img_size, img_size)


# ---------------------------------------------------------------
# RSNA-Kaggle 2.5D Model
# ---------------------------------------------------------------

class RSNAKaggle2p5D(nn.Module):
    """
    2.5D CNN + GRU classifier.

    Per-patient pipeline:
        [B, 1, H, W, S]
            -> resample to 96 slices
            -> 32 triplets of [3, H, W]        (2.5D grouping)
            -> shared 2D CNN backbone           (32 × forward passes)
            -> 32 × [D] feature vectors
            -> bidirectional GRU                (sequential, order matters)
            -> mean pool over 32 timesteps
            -> linear classifier
    """

    def __init__(
        self,
        num_classes:   int   = 4,
        backbone_name: str   = 'efficientnet_b0',
        target_slices: int   = 96,
        n_triplets:    int   = 32,
        img_size:      int   = 224,
        gru_hidden:    int   = 256,
        gru_layers:    int   = 2,
        bidirectional: bool  = True,
        dropout:       float = 0.2,
    ):
        super().__init__()
        self.target_slices = target_slices
        self.n_triplets    = n_triplets
        self.img_size      = img_size

        # 2D CNN backbone — shared weights across all 32 triplets
        print(f"Loading CNN backbone: {backbone_name}...")
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,     # remove head, output feature vector
            in_chans=3,
        )
        cnn_out_dim = self.backbone.num_features
        print(f"CNN output dim: {cnn_out_dim}")

        # GRU — processes the 32 triplet features in anatomical order
        self.gru = nn.GRU(
            input_size=cnn_out_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        gru_out_dim = gru_hidden * (2 if bidirectional else 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, num_classes),
        )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nRSNA-Kaggle 2.5D initialized:")
        print(f"  Backbone:         {backbone_name} (out_dim={cnn_out_dim})")
        print(f"  GRU:              {gru_layers}L x {gru_hidden}H "
              f"({'bi' if bidirectional else 'uni'}directional)")
        print(f"  Trainable params: {n_trainable:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W, S] single-channel volume, values in [0, 1]
        Returns:
            logits: [B, num_classes]
        """
        B = x.shape[0]

        # 1. Resample to 96 slices
        x = resample_volume(x, target_slices=self.target_slices)   # [B, 1, H, W, 96]

        # 2. Form 32 triplets
        triplets = volume_to_triplets(x,
                                      n_triplets=self.n_triplets,
                                      img_size=self.img_size)      # [B, 32, 3, H, W]

        # 3. Encode each triplet with shared CNN
        triplets_flat = triplets.reshape(B * self.n_triplets, 3,
                                         self.img_size, self.img_size)
        feats_flat = self.backbone(triplets_flat)                   # [B*32, D]
        feats = feats_flat.reshape(B, self.n_triplets, -1)         # [B, 32, D]

        # 4. GRU over the 32-step sequence
        gru_out, _ = self.gru(feats)                               # [B, 32, gru_out_dim]

        # 5. Mean pool and classify
        pooled = gru_out.mean(dim=1)                               # [B, gru_out_dim]
        return self.classifier(pooled)                             # [B, num_classes]


# ---------------------------------------------------------------
# PyTorch Lightning Module
# ---------------------------------------------------------------

class RSNAKaggleLightningModule(L.LightningModule):

    def __init__(
        self,
        num_classes:   int   = 4,
        backbone_name: str   = 'efficientnet_b0',
        target_slices: int   = 96,
        n_triplets:    int   = 32,
        img_size:      int   = 224,
        gru_hidden:    int   = 256,
        gru_layers:    int   = 2,
        bidirectional: bool  = True,
        dropout:       float = 0.2,
        lr:            float = 1e-4,
        weight_decay:  float = 1e-4,
        warmup_epochs: int   = 10,
        max_epochs:    int   = 150,
        focal_gamma:   float = 2.0,
        focal_alpha:   float = 0.25,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = RSNAKaggle2p5D(
            num_classes=num_classes,
            backbone_name=backbone_name,
            target_slices=target_slices,
            n_triplets=n_triplets,
            img_size=img_size,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )

        metric_kwargs = dict(task='multiclass', num_classes=num_classes)
        self.val_auroc   = AUROC(**metric_kwargs)
        self.val_acc     = Accuracy(**metric_kwargs)
        self.val_f1      = F1Score(average='macro', **metric_kwargs)
        self.val_bal_acc = BalancedAccuracy(num_classes=num_classes)

        self.test_auroc   = AUROC(**metric_kwargs)
        self.test_acc     = Accuracy(**metric_kwargs)
        self.test_f1      = F1Score(average='macro', **metric_kwargs)
        self.test_bal_acc = BalancedAccuracy(num_classes=num_classes)

    def focal_loss(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return (self.hparams.focal_alpha * (1 - pt) ** self.hparams.focal_gamma * ce).mean()

    def _unpack_batch(self, batch):
        x, y, *_ = batch
        return x, y

    def _shared_step(self, batch):
        x, y = self._unpack_batch(batch)
        logits = self.model(x)
        loss = self.focal_loss(logits, y)
        probs = F.softmax(logits, dim=1)
        return loss, probs, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch)
        preds = probs.argmax(dim=1)
        self.val_auroc.update(probs, y)
        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)
        self.val_bal_acc.update(preds, y)
        self.log('val/loss', loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log('val/AUROC',             self.val_auroc.compute(),   prog_bar=True)
        self.log('val/accuracy',          self.val_acc.compute(),     prog_bar=True)
        self.log('val/F1_macro',          self.val_f1.compute(),      prog_bar=True)
        self.log('val/balanced_accuracy', self.val_bal_acc.compute(), prog_bar=True)
        self.val_auroc.reset(); self.val_acc.reset()
        self.val_f1.reset();    self.val_bal_acc.reset()

    def test_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch)
        preds = probs.argmax(dim=1)
        self.test_auroc.update(probs, y)
        self.test_acc.update(preds, y)
        self.test_f1.update(preds, y)
        self.test_bal_acc.update(preds, y)

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
        x, y, *_ = batch
        logits = self.model(x)
        return y, logits

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                          total_iters=self.hparams.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer,
                                   T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
                                   eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine],
                                 milestones=[self.hparams.warmup_epochs])
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}


# ---------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------
if __name__ == '__main__':
    print("Running RSNA-Kaggle 2.5D sanity check...\n")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    B = 2
    x = torch.randn(B, 1, 64, 64, 64).clamp(0, 1).to(device)
    model = RSNAKaggle2p5D(num_classes=4, backbone_name='efficientnet_b0').to(device)

    with torch.no_grad():
        logits = model(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {logits.shape}")   # [2, 4]
    assert logits.shape == (B, 4)
    print("\nSanity check passed!")
