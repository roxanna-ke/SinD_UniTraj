#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import tempfile
from importlib import import_module
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNITRAJ_ROOT = PROJECT_ROOT / "UniTraj"
SCENARIONET_ROOT = PROJECT_ROOT / "scenarionet"

for path in (PROJECT_ROOT, UNITRAJ_ROOT, SCENARIONET_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

METHODS = {
    "MTR": {
        "config": "MTR.yaml",
        "dataset_module": "unitraj.datasets.MTR_dataset",
        "dataset_class": "MTRDataset",
    },
    "autobot": {
        "config": "autobot.yaml",
        "dataset_module": "unitraj.datasets.autobot_dataset",
        "dataset_class": "AutoBotDataset",
    },
    "wayformer": {
        "config": "wayformer.yaml",
        "dataset_module": "unitraj.datasets.wayformer_dataset",
        "dataset_class": "WayformerDataset",
    },
}


def _dataset_class(method: str):
    spec = METHODS[method]
    module = import_module(spec["dataset_module"])
    return getattr(module, spec["dataset_class"])


def _build_cfg(config_path: Path, unitraj_root: Path, method: str, split_dir: Path, cache_root: Path, overwrite: bool):
    from omegaconf import OmegaConf

    method_path = unitraj_root / "unitraj" / "configs" / "method" / METHODS[method]["config"]
    if not method_path.exists():
        raise FileNotFoundError(f"Missing method config: {method_path}")

    cfg = OmegaConf.load(config_path)
    method_cfg = OmegaConf.load(method_path)
    OmegaConf.set_struct(cfg, False)
    cfg = OmegaConf.merge(cfg, {"method": method_cfg})
    cfg = OmegaConf.merge(cfg, cfg.method)
    cfg.train_data_path = [str(split_dir)]
    cfg.val_data_path = [str(split_dir)]
    cfg.cache_path = str(cache_root)
    cfg.max_data_num = [None]
    cfg.starting_frame = [0]
    cfg.use_cache = False
    cfg.overwrite_cache = overwrite
    cfg.store_data_in_memory = False
    return cfg


def _assert_finite_array(value, label: str) -> None:
    if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
        raise ValueError(f"{label} contains NaN or Inf")


def _validate_scenario(split_dir: Path, mapping: dict, scenario_file: str, total_steps: int) -> None:
    from scenarionet.common_utils import read_scenario

    scenario = read_scenario(str(split_dir), mapping, scenario_file)
    for key in ("tracks", "dynamic_map_states", "map_features", "metadata"):
        if key not in scenario:
            raise ValueError(f"{scenario_file} missing required top-level key: {key}")

    tracks = scenario["tracks"]
    if not tracks:
        raise ValueError(f"{scenario_file} has no tracks")

    for track_id, track in tracks.items():
        if "state" not in track or "type" not in track:
            raise ValueError(f"{scenario_file} track {track_id} missing state/type")
        state = track["state"]
        for key in ("position", "length", "width", "height", "heading", "velocity", "valid"):
            if key not in state:
                raise ValueError(f"{scenario_file} track {track_id} missing state.{key}")
            value = np.asarray(state[key])
            if value.shape[0] < total_steps:
                raise ValueError(f"{scenario_file} track {track_id} state.{key} has {value.shape[0]} steps, expected >= {total_steps}")
            _assert_finite_array(value, f"{scenario_file} track {track_id} state.{key}")

    ts = np.asarray(scenario["metadata"].get("ts", []))
    if ts.shape[0] < total_steps:
        raise ValueError(f"{scenario_file} metadata.ts has {ts.shape[0]} steps, expected >= {total_steps}")


def _load_and_validate_split(split_dir: Path, total_steps: int, sample_count: int) -> tuple[dict, list[str], dict]:
    from scenarionet.common_utils import read_dataset_summary

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
    for file_name in ("dataset_summary.pkl", "dataset_mapping.pkl"):
        if not (split_dir / file_name).exists():
            raise FileNotFoundError(f"Missing {file_name} in {split_dir}")

    summary, scenario_files, mapping = read_dataset_summary(str(split_dir))
    if not scenario_files:
        raise ValueError(f"No scenarios found in {split_dir}")

    for scenario_file in scenario_files[:sample_count]:
        if scenario_file not in summary:
            raise ValueError(f"{scenario_file} missing from dataset_summary.pkl")
        if scenario_file not in mapping:
            raise ValueError(f"{scenario_file} missing from dataset_mapping.pkl")
        scenario_path = split_dir / mapping[scenario_file] / scenario_file
        if not scenario_path.exists():
            raise FileNotFoundError(f"Mapped scenario file does not exist: {scenario_path}")
        _validate_scenario(split_dir, mapping, scenario_file, total_steps)

    return summary, list(scenario_files), mapping


def _write_subset(source_dir: Path, output_dir: Path, scenario_files: list[str], summary: dict, mapping: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_summary = {}
    subset_mapping = {}
    for scenario_file in scenario_files:
        rel_dir = mapping[scenario_file]
        src = source_dir / rel_dir / scenario_file
        dst_dir = output_dir / rel_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / scenario_file)
        subset_summary[scenario_file] = summary[scenario_file]
        subset_mapping[scenario_file] = rel_dir

    with (output_dir / "dataset_summary.pkl").open("wb") as f:
        pickle.dump(subset_summary, f)
    with (output_dir / "dataset_mapping.pkl").open("wb") as f:
        pickle.dump(subset_mapping, f)


def _dry_run_unitraj(method: str, split_dir: Path, summary: dict, scenario_files: list[str], mapping: dict, args: argparse.Namespace) -> None:
    if args.dry_run_samples <= 0:
        return

    phase = split_dir.parent.name
    dataset_name = split_dir.name
    sample_count = min(len(scenario_files), max(args.dry_run_samples, min(32, len(scenario_files))))
    sample_files = scenario_files[:sample_count]
    Dataset = _dataset_class(method)

    with tempfile.TemporaryDirectory(prefix=f"sind_unitraj_{method}_") as tmp:
        tmp_path = Path(tmp)
        subset_dir = tmp_path / "subset" / phase / dataset_name
        _write_subset(split_dir, subset_dir, sample_files, summary, mapping)
        cfg = _build_cfg(args.config, args.unitraj_root, method, subset_dir, tmp_path / "cache" / method, overwrite=True)

        old_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            dataset = Dataset(cfg, is_validation=False)
            if len(dataset) == 0:
                raise ValueError(
                    f"UniTraj dry-run produced 0 samples for {method} on {split_dir}; "
                    f"tried {len(sample_files)} of {len(scenario_files)} scenarios. "
                    "Check earlier 'Warning:' lines from BaseDataset for the preprocessing reason."
                )
            first = dataset[0]
            for key in ("obj_trajs", "obj_trajs_mask", "map_polylines", "track_index_to_predict"):
                if key not in first:
                    raise ValueError(f"UniTraj dry-run output for {method} missing key: {key}")
        finally:
            os.chdir(old_cwd)


def _build_cache(method: str, split_dir: Path, cache_root: Path, args: argparse.Namespace) -> None:
    Dataset = _dataset_class(method)
    cfg = _build_cfg(args.config, args.unitraj_root, method, split_dir, cache_root, overwrite=args.overwrite)
    Dataset(cfg, is_validation=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate SinD ScenarioNet splits and build UniTraj caches for MTR, AutoBot, and Wayformer.")
    parser.add_argument("--train-dir", type=Path, required=True, help="ScenarioNet train split, e.g. .../record_level/train/sind")
    parser.add_argument("--val-dir", type=Path, required=True, help="ScenarioNet val/test split, e.g. .../record_level/test/sind")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "UniTraj" / "cache" / "sind", help="Output root. Each method gets its own subdirectory.")
    parser.add_argument("--unitraj-root", type=Path, default=UNITRAJ_ROOT)
    parser.add_argument("--config", type=Path, default=UNITRAJ_ROOT / "unitraj" / "configs" / "config.yaml")
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--check-samples", type=int, default=8, help="Number of raw scenarios to structurally validate per split.")
    parser.add_argument("--dry-run-samples", type=int, default=2, help="Number of scenarios to run through each UniTraj Dataset before full cache build.")
    parser.add_argument("--past-len", type=int, default=21)
    parser.add_argument("--future-len", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing UniTraj cache directories.")
    parser.add_argument("--validate-only", action="store_true", help="Run checks but do not build full caches.")
    args = parser.parse_args()

    total_steps = args.past_len + args.future_len
    splits = {"train": args.train_dir, "val": args.val_dir}

    validated = {}
    for split_name, split_dir in splits.items():
        summary, scenario_files, mapping = _load_and_validate_split(split_dir, total_steps, args.check_samples)
        validated[split_name] = (summary, scenario_files, mapping)
        print(f"[ok] {split_name}: {len(scenario_files)} scenarios passed structural checks in {split_dir}", flush=True)

    for method in args.methods:
        for split_name, split_dir in splits.items():
            summary, scenario_files, mapping = validated[split_name]
            print(f"[check] {method} {split_name}: UniTraj dry-run on {min(args.dry_run_samples, len(scenario_files))} scenarios", flush=True)
            _dry_run_unitraj(method, split_dir, summary, scenario_files, mapping, args)

    if args.validate_only:
        print("[done] validation passed; full cache build skipped because --validate-only was set", flush=True)
        return

    for method in args.methods:
        method_cache_root = args.cache_root / method
        for split_name, split_dir in splits.items():
            print(f"[build] {method} {split_name}: cache root {method_cache_root}", flush=True)
            _build_cache(method, split_dir, method_cache_root, args)

    print(f"[done] caches written under {args.cache_root}", flush=True)


if __name__ == "__main__":
    main()
