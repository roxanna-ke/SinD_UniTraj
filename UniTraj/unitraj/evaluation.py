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
from unitraj.utils.utils import set_seed
import hydra
from omegaconf import OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="config")
def evaluation(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)  # Open the struct
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg['eval'] = True

    model = build_model(cfg)

    val_set = build_dataset(cfg, val=True)

    eval_batch_size = cfg.method['eval_batch_size']

    val_loader = DataLoader(
        val_set, batch_size=eval_batch_size, num_workers=cfg.load_num_workers, shuffle=False, drop_last=False,
        collate_fn=val_set.collate_fn)

    wandb_project = os.environ.get("WANDB_PROJECT", cfg.get("wandb_project", "SinD_UniTraj"))

    trainer = pl.Trainer(
        inference_mode=True,
        logger=None if cfg.debug else WandbLogger(project=wandb_project, name=cfg.exp_name),
        devices=1,
        accelerator="cpu" if cfg.debug else "gpu",
        profiler="simple",
    )

    trainer.validate(model=model, dataloaders=val_loader, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    evaluation()
