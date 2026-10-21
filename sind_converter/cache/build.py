from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf


METHOD_CONFIGS = {"autobot": "autobot.yaml", "mtr": "MTR.yaml", "wayformer": "wayformer.yaml"}


def build_unitraj_cache(
    split_dataset_dir: Path,
    unitraj_root: Path,
    config_path: Path,
    method: str = "autobot",
    cache_root: Path | None = None,
    use_traffic_light_tokens: bool = False,
) -> None:
    method = method.lower()
    if method not in METHOD_CONFIGS:
        raise ValueError(f"Unsupported UniTraj method: {method}")

    unitraj_root = Path(unitraj_root).resolve()
    scenarionet_root = Path(__file__).resolve().parents[2] / "scenarionet"
    for root in (unitraj_root, scenarionet_root):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    method_path = unitraj_root / "unitraj" / "configs" / "method" / METHOD_CONFIGS[method]
    cfg = OmegaConf.load(config_path)
    method_cfg = OmegaConf.load(method_path)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, {"method": method_cfg})
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg.train_data_path = [str(split_dataset_dir)]
    cfg.val_data_path = [str(split_dataset_dir)]
    cfg.max_data_num = [None]
    cfg.starting_frame = [0]
    cfg.use_cache = False
    cfg.overwrite_cache = True
    cfg.use_traffic_light_tokens = use_traffic_light_tokens
    if cache_root is not None:
        cfg.cache_path = str(cache_root)

    if method == "autobot":
        from unitraj.datasets.autobot_dataset import AutoBotDataset as Dataset
    elif method == "mtr":
        from unitraj.datasets.MTR_dataset import MTRDataset as Dataset
    else:
        from unitraj.datasets.wayformer_dataset import WayformerDataset as Dataset
    Dataset(cfg, is_validation=False)
