from __future__ import annotations

import json
import pickle
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIONET_ROOT = PROJECT_ROOT / "scenarionet"
if SCENARIONET_ROOT.exists() and str(SCENARIONET_ROOT) not in sys.path:
    sys.path.insert(0, str(SCENARIONET_ROOT))

from scenarionet.common_utils import read_dataset_summary


def _write_subset(canonical_root: Path, output_dir: Path, scenario_files: list[str], summary: dict, mapping: dict) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_summary = {}
    subset_mapping = {}
    for file_name in scenario_files:
        rel_dir = mapping[file_name]
        src = canonical_root / rel_dir / file_name
        dst_dir = output_dir / rel_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / file_name)
        subset_summary[file_name] = summary[file_name]
        subset_mapping[file_name] = rel_dir
    with (output_dir / "dataset_summary.pkl").open("wb") as f:
        pickle.dump(subset_summary, f)
    with (output_dir / "dataset_mapping.pkl").open("wb") as f:
        pickle.dump(subset_mapping, f)
    return output_dir


def _record_name(meta: dict) -> str:
    return str(meta.get("record_name") or meta.get("source_file") or meta.get("split_candidates", {}).get("record"))


def _city_name(meta: dict) -> str:
    return str(meta.get("city") or meta.get("split_candidates", {}).get("city"))


def _write_assignment(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "generated_at": datetime.now(timezone.utc).isoformat(), "config_version": "sind_converter_v1"}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def make_record_level_split(
    canonical_root: Path,
    output_root: Path,
    dataset_name: str = "sind",
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict[str, Path]:
    summary, scenario_files, mapping = read_dataset_summary(canonical_root)
    records = sorted({_record_name(summary[file_name]) for file_name in scenario_files})
    rng = random.Random(seed)
    rng.shuffle(records)
    split_index = max(1, min(len(records) - 1, int(len(records) * train_ratio))) if len(records) > 1 else len(records)
    train_records = set(records[:split_index])
    test_records = set(records[split_index:])
    train_files = [file_name for file_name in scenario_files if _record_name(summary[file_name]) in train_records]
    test_files = [file_name for file_name in scenario_files if _record_name(summary[file_name]) in test_records]

    train_dir = Path(output_root) / "record_level" / "train" / dataset_name
    test_dir = Path(output_root) / "record_level" / "test" / dataset_name
    _write_subset(Path(canonical_root), train_dir, train_files, summary, mapping)
    _write_subset(Path(canonical_root), test_dir, test_files, summary, mapping)
    _write_assignment(
        Path(output_root) / "record_level" / "split_assignment.json",
        {"mode": "record_level", "seed": seed, "train_ratio": train_ratio, "train_records": sorted(train_records), "test_records": sorted(test_records)},
    )
    return {"train": train_dir, "test": test_dir}


def make_city_holdout_split(
    canonical_root: Path,
    output_root: Path,
    heldout_cities: list[str] | None = None,
    train_cities: list[str] | None = None,
    test_cities: list[str] | None = None,
    dataset_name: str = "sind",
) -> dict[str, Path]:
    summary, scenario_files, mapping = read_dataset_summary(canonical_root)
    all_cities = sorted({_city_name(summary[file_name]) for file_name in scenario_files})
    if heldout_cities is not None:
        test_city_set = set(heldout_cities)
        train_city_set = set(all_cities) - test_city_set
    else:
        train_city_set = set(train_cities or [])
        test_city_set = set(test_cities or [])
    if not train_city_set or not test_city_set:
        raise ValueError("City-level split requires non-empty train and test city sets")

    train_files = [file_name for file_name in scenario_files if _city_name(summary[file_name]) in train_city_set]
    test_files = [file_name for file_name in scenario_files if _city_name(summary[file_name]) in test_city_set]
    train_dir = Path(output_root) / "city_holdout" / "train" / dataset_name
    test_dir = Path(output_root) / "city_holdout" / "test" / dataset_name
    _write_subset(Path(canonical_root), train_dir, train_files, summary, mapping)
    _write_subset(Path(canonical_root), test_dir, test_files, summary, mapping)
    _write_assignment(
        Path(output_root) / "city_holdout" / "split_assignment.json",
        {"mode": "city_holdout", "train_cities": sorted(train_city_set), "test_cities": sorted(test_city_set)},
    )
    return {"train": train_dir, "test": test_dir}
