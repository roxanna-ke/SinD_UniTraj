import os
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import matplotlib
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unitraj.datasets import build_dataset
from unitraj.models import build_model
from unitraj.utils import visualization
from unitraj.utils.utils import set_seed


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected checkpoint keys: {len(unexpected)}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def visualize(cfg):
    set_seed(cfg.seed)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, cfg.method)

    if not cfg.get("ckpt_path"):
        raise ValueError("Set ckpt_path=/path/to/model.ckpt")

    output_dir = Path(cfg.get("visualization_output_dir", "prediction_visualizations"))
    output_dir.mkdir(parents=True, exist_ok=True)
    num_images = int(cfg.get("num_prediction_visualizations", 32))

    device_name = cfg.get("visualization_device", None)
    if device_name is None:
        device_name = "cuda:0" if torch.cuda.is_available() and not cfg.debug else "cpu"
    device = torch.device(device_name)

    dataset = build_dataset(cfg, val=True)
    batch_size = int(cfg.get("visualization_batch_size", min(cfg.method.get("eval_batch_size", 16), 16)))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=cfg.load_num_workers,
        shuffle=False,
        drop_last=False,
        collate_fn=dataset.collate_fn,
    )

    model = build_model(cfg)
    _load_checkpoint(model, cfg.ckpt_path)
    model.to(device)
    model.eval()

    saved = 0
    with torch.inference_mode():
        for batch_idx, batch in enumerate(dataloader):
            batch = _move_to_device(batch, device)
            prediction, _ = model(batch)
            batch_size_actual = int(batch["batch_size"])

            for draw_index in range(batch_size_actual):
                if saved >= num_images:
                    print(f"[done] saved {saved} figures to {output_dir}")
                    return

                scenario_id = str(batch["input_dict"]["scenario_id"][draw_index])
                object_id = str(batch["input_dict"]["center_objects_id"][draw_index])
                safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in scenario_id)
                output_path = output_dir / f"{saved:04d}_{safe_id}_{object_id}.png"

                plot = visualization.visualize_prediction(batch, prediction, draw_index=draw_index)
                plot.savefig(output_path, dpi=180, bbox_inches="tight", pad_inches=0.05)
                plot.close()
                plt.close("all")
                saved += 1

    print(f"[done] saved {saved} figures to {output_dir}")


if __name__ == "__main__":
    visualize()
