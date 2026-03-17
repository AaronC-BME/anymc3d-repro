"""
AnyMC3D Training Script — Hydra config driven

Updated to pass patch_size and percentile normalization params
to the refactored PDCADDataModule.

Usage:
    export CUDA_VISIBLE_DEVICES=2
    python train.py                                          # AnyMC3D ViT-L (default)
    python train.py model=anymc3d_vitb                      # AnyMC3D ViT-B
    python train.py model=rsna_effb0                        # RSNA CNN EfficientNet-B0
    python train.py model=rsna_effb4                        # RSNA CNN EfficientNet-B4
    python train.py model=rsna_effb0 data.fold=1            # different fold
    python train.py model=anymc3d_vitl model.run_name=myrun # override run name
"""

import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import wandb
from pathlib import Path


def get_datamodule(cfg, fold: int):
    if cfg.data.dataset == "pdcad":
        from pdcad_dataset import PDCADDataModule

        # Extract patch_size from config if available
        # pdcad.yaml: train.patch_size: [308, 308, 70]
        patch_size = None
        if hasattr(cfg, 'train') and hasattr(cfg.train, 'patch_size'):
            patch_size = list(cfg.train.patch_size)
        elif hasattr(cfg.data, 'patch_size'):
            patch_size = list(cfg.data.patch_size)

        return PDCADDataModule(
            data_root   = cfg.data.data_root,
            labels_path = cfg.data.labels_path,
            splits_path = cfg.data.splits_path,
            fold        = fold,
            batch_size  = cfg.data.batch_size,
            num_workers = cfg.data.num_workers,
            augment     = cfg.data.augment,
            patch_size  = patch_size,          # [308, 308, 70] from config
        )
    elif cfg.data.dataset == "meningioma_t1c":
        from meningioma_t1c_dataset import MeningiomaT1cDataModule
        return MeningiomaT1cDataModule(
            data_root   = cfg.data.data_root,
            labels_path = cfg.data.labels_path,
            splits_path = cfg.data.splits_path,
            fold        = fold,
            batch_size  = cfg.data.batch_size,
            num_workers = cfg.data.num_workers,
            augment     = cfg.data.augment,
        )
    else:
        from meningioma_holdout_dataset import MeningiomaDataModule
        return MeningiomaDataModule(
            data_root   = cfg.data.data_root,
            labels_path = cfg.data.labels_path,
            splits_path = cfg.data.splits_path,
            fold        = fold,
            batch_size  = cfg.data.batch_size,
            num_workers = cfg.data.num_workers,
            augment     = cfg.data.augment,
        )


def get_model(cfg):
    if cfg.model.arch == "anymc3d":
        from anymc3d import AnyMC3DLightningModule
        return AnyMC3DLightningModule(
            num_classes       = cfg.model.num_classes,
            modalities        = list(cfg.model.modalities),
            backbone_name     = cfg.model.backbone_name,
            lora_rank         = cfg.model.lora_rank,
            lora_alpha        = cfg.model.lora_alpha,
            input_size        = cfg.model.input_size,
            dropout           = cfg.model.dropout,
            slice_axis        = cfg.model.slice_axis,
            lora_lr           = cfg.optimizer.lora_lr,
            lora_weight_decay = cfg.optimizer.lora_weight_decay,
            head_lr           = cfg.optimizer.head_lr,
            head_weight_decay = cfg.optimizer.head_weight_decay,
            warmup_epochs     = cfg.optimizer.warmup_epochs,
            lr_scheduler      = cfg.optimizer.lr_scheduler,
            focal_gamma       = cfg.loss.focal_gamma,
            focal_alpha       = cfg.loss.focal_alpha,
            max_epochs        = cfg.training.max_epochs,
        )
    elif cfg.model.arch == "rsna_cnn":
        from rsna_kaggle_model import RSNAKaggleLightningModule
        return RSNAKaggleLightningModule(
            num_classes   = cfg.model.num_classes,
            backbone_name = cfg.model.backbone_name,
            target_slices = cfg.model.target_slices,
            n_triplets    = cfg.model.n_triplets,
            img_size      = cfg.model.img_size,
            gru_hidden    = cfg.model.gru_hidden,
            gru_layers    = cfg.model.gru_layers,
            bidirectional = cfg.model.bidirectional,
            dropout       = cfg.model.dropout,
            lr            = cfg.optimizer.lr,
            weight_decay  = cfg.optimizer.weight_decay,
            warmup_epochs = cfg.optimizer.warmup_epochs,
            lr_scheduler  = cfg.optimizer.lr_scheduler,
            focal_gamma   = cfg.loss.focal_gamma,
            focal_alpha   = cfg.loss.focal_alpha,
            max_epochs    = cfg.training.max_epochs,
        )
    else:
        raise ValueError(f"Unknown arch '{cfg.model.arch}'. Choose from: anymc3d, rsna_cnn")


def train_one_fold(cfg, fold: int, multi_fold: bool = False):
    base_name = cfg.model.run_name
    if multi_fold:
        run_name = base_name
        ckpt_dir = Path("checkpoints") / base_name / str(fold)
    else:
        run_name = f"{base_name}_fold{fold}"
        ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    wandb_run_name = f"{base_name}_fold{fold}"

    print(f"\n{'='*60}")
    print(f"  Fold {fold} — {wandb_run_name}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    print(f"{'='*60}")

    OmegaConf.save(cfg, ckpt_dir / "config.yaml")
    print(f"Config saved -> {ckpt_dir / 'config.yaml'}\n")

    dm    = get_datamodule(cfg, fold)
    model = get_model(cfg)

    wandb_logger = WandbLogger(
        project = cfg.wandb.project,
        name    = wandb_run_name,
        config  = OmegaConf.to_container(cfg, resolve=True),
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath    = str(ckpt_dir),
        monitor    = "val/AUROC",
        mode       = "max",
        save_top_k = cfg.training.save_top_k,
        filename   = "epoch={epoch:02d}-val_auroc={val/AUROC:.4f}",
        auto_insert_metric_name = False,
        verbose    = True,
    )
    early_stop_cb = EarlyStopping(
        monitor  = "val/loss",
        mode     = "min",
        patience = cfg.training.early_stopping_patience,
        verbose  = True,
    )
    lr_monitor = LearningRateMonitor()

    trainer = L.Trainer(
        max_epochs        = cfg.training.max_epochs,
        accelerator       = "gpu",
        devices           = 1,
        precision         = cfg.training.precision,
        logger            = wandb_logger,
        callbacks         = [checkpoint_cb, early_stop_cb, lr_monitor],
        log_every_n_steps = cfg.training.log_every_n_steps,
        deterministic     = False,
    )

    print(f"Starting training — fold {fold}...")
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    print(f"\nRunning test set evaluation — fold {fold}...")
    trainer.test(model, dm.test_dataloader(), ckpt_path="best")

    wandb.finish()
    print(f"\nFold {fold} complete. Outputs: {ckpt_dir}")


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:

    torch.set_float32_matmul_precision('medium')
    L.seed_everything(cfg.training.seed)

    print("\n" + "=" * 60)
    print(f"Training: {cfg.model.run_name}")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60 + "\n")

    # ── Determine folds to run ────────────────────────────────────────────────
    fold_cfg = cfg.data.fold
    if isinstance(fold_cfg, (list, ListConfig)):
        folds = list(fold_cfg)
        print(f"Multi-fold training: folds {folds}\n")
    else:
        folds = [int(fold_cfg)]
        print(f"Single-fold training: fold {folds[0]}\n")

    # ── Run folds sequentially ────────────────────────────────────────────────
    multi_fold = len(folds) > 1
    for fold in folds:
        train_one_fold(cfg, fold, multi_fold=multi_fold)

    print(f"\n{'='*60}")
    print(f"All folds complete: {folds}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()