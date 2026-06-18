import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytorch_lightning as pl
import torch

torch.set_float32_matmul_precision('medium')
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from unitraj.models import build_model
from unitraj.datasets import build_dataset
from unitraj.utils.utils import set_seed, find_latest_checkpoint
from pytorch_lightning.callbacks import ModelCheckpoint  # Import ModelCheckpoint
import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)

    model = build_model(cfg)

    train_set = build_dataset(cfg)
    val_set = build_dataset(cfg, val=True)

    train_batch_size = max(cfg.method['train_batch_size'] // len(cfg.devices),  1)
    eval_batch_size = max(cfg.method['eval_batch_size'] // len(cfg.devices), 1)

    call_backs = []

    checkpoint_callback = ModelCheckpoint(
        monitor='val/brier_fde',  # Replace with your validation metric
        filename='{epoch}-{val/brier_fde:.2f}',
        save_top_k=1,
        mode='min',  # 'min' for loss/error, 'max' for accuracy
        dirpath=f'./unitraj_ckpt/{cfg.exp_name}'
    )

    call_backs.append(checkpoint_callback)

    train_loader = DataLoader(
        train_set, batch_size=train_batch_size, num_workers=cfg.load_num_workers, drop_last=False,
        collate_fn=train_set.collate_fn)

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=train_set.collate_fn)

    use_ddp = (not cfg.debug) and len(cfg.devices) > 1

    wandb_project = os.environ.get("WANDB_PROJECT", cfg.get("wandb_project", "SinD_UniTraj"))
    logger = None if cfg.debug else WandbLogger(project=wandb_project, name=cfg.exp_name, id=cfg.exp_name)

    trainer = pl.Trainer(
        max_epochs=cfg.method.max_epochs,
        logger=logger,
        devices=1 if cfg.debug else cfg.devices,
        gradient_clip_val=cfg.method.grad_clip_norm,
        # accumulate_grad_batches=cfg.method.Trainer.accumulate_grad_batches,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
        strategy="ddp" if use_ddp else "auto",
        callbacks=call_backs,
        log_every_n_steps=1,
        limit_train_batches=cfg.get('limit_train_batches', 1.0),
        limit_val_batches=cfg.get('limit_val_batches', 1.0),
        num_sanity_val_steps=cfg.get('num_sanity_val_steps', 2),
    )

    # automatically resume training
    if cfg.ckpt_path is None and not cfg.debug:
        # Pattern to match all .ckpt files in the base_path recursively
        search_pattern = os.path.join('./unitraj_ckpt', cfg.exp_name, '**', '*.ckpt')
        cfg.ckpt_path = find_latest_checkpoint(search_pattern)

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path)

    if cfg.get('full_validate_after_fit', False):
        validate_ckpt_path = checkpoint_callback.best_model_path or cfg.ckpt_path
        print(f"Running full validation after fit with checkpoint: {validate_ckpt_path or 'current model'}")
        full_val_trainer = pl.Trainer(
            logger=logger,
            devices=1 if cfg.debug else cfg.devices,
            accelerator="cpu" if cfg.debug else "gpu",
            strategy="ddp" if use_ddp else "auto",
            limit_val_batches=cfg.get('full_validate_limit_val_batches', 1.0),
            num_sanity_val_steps=0,
            log_every_n_steps=1,
        )
        full_val_trainer.validate(model=model, dataloaders=val_loader, ckpt_path=validate_ckpt_path or None)


if __name__ == '__main__':
    train()
