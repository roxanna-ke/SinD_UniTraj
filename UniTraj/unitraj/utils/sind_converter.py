from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sind_converter.config.defaults import ConverterConfig
from sind_converter.data.discovery import RecordDescription, discover_records, resolve_map_path
from sind_converter.scenarios.convert import _write_scenarios, convert_scenarios, windows_for_record


def _default_map_fallback(record_dir: Path) -> Path:
    for parent in [record_dir, *record_dir.parents]:
        candidate = parent / "SinD" / "Data"
        if candidate.exists():
            return candidate
        if parent.name in {"Dataset", "Data"}:
            data_candidate = parent.parent / "Data"
            if data_candidate.exists():
                return data_candidate
    return record_dir.parent


def convert_sind_record(
    sind_record_dir: Path,
    output_dir: Path,
    city: str,
    dataset_name: str = "sind",
    dataset_version: str = "v1",
    past_len: int = 21,
    future_len: int = 60,
    stride: int = 40,
    max_scenarios: int | None = None,
    train_ratio: float = 0.8,
) -> dict[str, Path]:
    sind_record_dir = Path(sind_record_dir)
    data_root = sind_record_dir.parents[1]
    map_fallback_root = _default_map_fallback(sind_record_dir)
    map_path = resolve_map_path(city, data_root, map_fallback_root)
    traffic_light = next(iter(sorted(sind_record_dir.glob("Traffic*.csv"))), None)
    record = RecordDescription(
        city=city,
        record_name=sind_record_dir.name,
        record_dir=sind_record_dir,
        vehicle_tracks_path=sind_record_dir / "Veh_smoothed_tracks.csv",
        pedestrian_tracks_path=sind_record_dir / "Ped_smoothed_tracks.csv",
        traffic_light_path=traffic_light,
        vehicle_meta_path=sind_record_dir / "Veh_tracks_meta.csv" if (sind_record_dir / "Veh_tracks_meta.csv").exists() else None,
        pedestrian_meta_path=sind_record_dir / "Ped_tracks_meta.csv" if (sind_record_dir / "Ped_tracks_meta.csv").exists() else None,
        recording_meta_path=next(
            (path for path in [sind_record_dir / "recording_metas.csv", sind_record_dir / "recoding_metas.csv"] if path.exists()),
            None,
        ),
        map_path=map_path,
    )
    windows = windows_for_record(record, past_len=past_len, future_len=future_len, stride=stride, max_scenarios=max_scenarios)
    if len(windows) < 2:
        raise ValueError("Not enough eligible SinD windows were generated")
    split_index = max(1, min(len(windows) - 1, int(len(windows) * train_ratio)))
    train_dir = Path(output_dir) / "train" / dataset_name
    val_dir = Path(output_dir) / "val" / dataset_name
    _write_scenarios(windows[:split_index], train_dir, dataset_name=dataset_name, dataset_version=dataset_version)
    _write_scenarios(windows[split_index:], val_dir, dataset_name=dataset_name, dataset_version=dataset_version)
    return {"train": train_dir, "val": val_dir}


def convert_sind_dataset(
    sind_data_root: Path,
    output_dir: Path,
    dataset_name: str = "sind",
    dataset_version: str = "v1",
    cities: list[str] | None = None,
    max_records: int | None = None,
    stride: int = 40,
    max_scenarios_per_record: int | None = None,
    train_ratio: float = 0.8,
) -> dict[str, Path]:
    data_root = Path(sind_data_root)
    map_fallback_root = data_root if data_root.name == "Data" else data_root.parent / "Data"
    records = discover_records(data_root, map_fallback_root, cities=cities)
    if max_records is not None:
        records = records[:max_records]
    records_with_windows = []
    for record in records:
        windows = windows_for_record(record, past_len=21, future_len=60, stride=stride, max_scenarios=max_scenarios_per_record)
        if windows:
            records_with_windows.append((record, windows))
    if len(records_with_windows) < 2:
        raise ValueError("Need at least two non-empty SinD records after window generation")
    split_index = max(1, min(len(records_with_windows) - 1, int(len(records_with_windows) * train_ratio)))
    train_windows = [window for _, windows in records_with_windows[:split_index] for window in windows]
    val_windows = [window for _, windows in records_with_windows[split_index:] for window in windows]
    train_dir = Path(output_dir) / "train" / dataset_name
    val_dir = Path(output_dir) / "val" / dataset_name
    _write_scenarios(train_windows, train_dir, dataset_name=dataset_name, dataset_version=dataset_version)
    _write_scenarios(val_windows, val_dir, dataset_name=dataset_name, dataset_version=dataset_version)
    return {"train": train_dir, "val": val_dir}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for the modular SinD converter")
    parser.add_argument("--sind-data-root", type=Path, required=True)
    parser.add_argument("--map-fallback-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-scenarios-per-record", type=int, default=None)
    args = parser.parse_args()
    config = ConverterConfig(
        data_root=args.sind_data_root,
        map_fallback_root=args.map_fallback_root,
        canonical_scenario_root=args.output_dir,
        split_root=args.output_dir / "splits",
        cache_root=args.output_dir / "cache",
        stride=args.stride,
    )
    convert_scenarios(config, cities=args.cities, max_records=args.max_records, max_scenarios_per_record=args.max_scenarios_per_record)


if __name__ == "__main__":
    main()
