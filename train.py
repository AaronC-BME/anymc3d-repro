"""
AnyMC3D Training Script — Hydra config driven

Augmentation is decoupled from data loading:
    get_datamodule(cfg, fold)  -> instantiates the data module via Hydra
    attach_augmentation(dm, …) -> transforms attached afterward

Layout:
    data_modules/pdcad_dataset.py       — PDCADDataModule
    data_modules/data_augmentation.py   — apply_augmentation, TransformedDataset
    model/anymc3d.py                    — AnyMC3DLightningModule

Usage:
    export CUDA_VISIBLE_DEVICES=2
    python train.py                                  # defaults (data=pdcad, model=anymc3d)
    python train.py data=pdcad model=anymc3d         # explicit
    python train.py data=meningioma_t1c              # swap data
    python train.py data.module.batch_size=4         # override a module kwarg
    python train.py 'data.module.fold=[0,1,2]'       # multi-fold

    nohup python train.py > logs/run.log 2>&1 &
"""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, ListConfig
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import wandb
from pathlib import Path


def get_datamodule(cfg, fold: int):
    """
    Build the raw data module — no augmentation attached here.

    For pdcad, the data module is instantiated directly from the config via
    hydra.utils.instantiate, using cfg.data.module (which carries _target_
    plus the constructor kwargs). The `fold` kwarg is passed as an explicit
    override so multi-fold loops pick up the current iteration's fold.
    """
    return instantiate(cfg.data.module, fold=fold)


def attach_augmentation(dm, cfg):
    """
    Attach train/eval transforms to the data module *after* construction.

    Reads cfg.data.augment (the flag, at the data-group level) — NOT
    cfg.data.module.augment, because `augment` isn't a PDCADDataModule
    constructor argument.
    """
    from data_modules.data_augmentation import apply_augmentation
    apply_augmentation(dm, augment_train=cfg.data.augment)
    # meningioma datasets still handle augmentation internally for now.
    return dm


def get_model(cfg):
    return instantiate(cfg.model)


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

    dm = get_datamodule(cfg, fold)

    attach_augmentation(dm, cfg)

    model = get_model(cfg)

    wandb_logger = WandbLogger(
        project = cfg.model.project,
        name    = wandb_run_name,
        config  = OmegaConf.to_container(cfg, resolve=True),
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath    = str(ckpt_dir),
        monitor    = "val/AUROC",
        mode       = "max",
        save_top_k = cfg.model.save_top_k,
        filename   = "epoch={epoch:02d}-val_auroc={val/AUROC:.4f}",
        auto_insert_metric_name = False,
        verbose    = True,
    )
    early_stop_cb = EarlyStopping(
        monitor  = "val/loss",
        mode     = "min",
        patience = cfg.model.early_stopping_patience,
        verbose  = True,
    )
    lr_monitor = LearningRateMonitor()

    trainer = L.Trainer(
        max_epochs        = cfg.model.max_epochs,
        accelerator       = "gpu",
        devices           = 1,
        precision         = cfg.model.precision,
        logger            = wandb_logger,
        callbacks         = [checkpoint_cb, early_stop_cb, lr_monitor],
        log_every_n_steps = cfg.model.log_every_n_steps,
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
    L.seed_everything(cfg.model.seed)

    print("\n" + "=" * 60)
    print(f"Training: {cfg.model.run_name}")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60 + "\n")

    fold_cfg = cfg.data.module.fold

    if isinstance(fold_cfg, (list, ListConfig)):
        folds = list(fold_cfg)
        print(f"Multi-fold training: folds {folds}\n")
    else:
        folds = [int(fold_cfg)]
        print(f"Single-fold training: fold {folds[0]}\n")

    multi_fold = len(folds) > 1
    for fold in folds:
        train_one_fold(cfg, fold, multi_fold=multi_fold)

    print(f"\n{'='*60}")
    print(f"All folds complete: {folds}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()