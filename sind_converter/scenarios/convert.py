from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIONET_ROOT = PROJECT_ROOT / "scenarionet"
if SCENARIONET_ROOT.exists() and str(SCENARIONET_ROOT) not in sys.path:
    sys.path.insert(0, str(SCENARIONET_ROOT))

from scenarionet.converter.utils import write_to_directory_single_worker

from sind_converter.config.defaults import ConverterConfig
from sind_converter.data.discovery import RecordDescription, discover_records
from sind_converter.data.loading import load_record
from sind_converter.maps.osm import parse_osm_map
from sind_converter.scenarios.build import ScenarioWindow, convert_window_to_scenario, generate_windows


def _write_scenarios(windows: list[ScenarioWindow], output_dir: Path, dataset_name: str, dataset_version: str) -> None:
    write_to_directory_single_worker(
        convert_func=convert_window_to_scenario,
        scenarios=windows,
        output_path=str(output_dir),
        dataset_version=dataset_version,
        dataset_name=dataset_name,
        overwrite=True,
        worker_index=0,
        report_memory_freq=None,
        preprocess=lambda scenarios, worker_index: scenarios,
    )


def windows_for_record(
    record: RecordDescription,
    past_len: int,
    future_len: int,
    stride: int,
    max_scenarios: int | None = None,
) -> list[ScenarioWindow]:
    loaded = load_record(record)
    map_features, lane_centers = parse_osm_map(record.map_path)
    return generate_windows(
        city=record.city,
        record_name=record.record_name,
        vehicle_tracks=loaded.vehicle_tracks,
        pedestrian_tracks=loaded.pedestrian_tracks,
        map_features=map_features,
        lane_centers=lane_centers,
        traffic_light=loaded.traffic_light,
        past_len=past_len,
        future_len=future_len,
        stride=stride,
        max_scenarios=max_scenarios,
    )


def convert_scenarios(
    config: ConverterConfig,
    cities: list[str] | None = None,
    max_records: int | None = None,
    max_scenarios_per_record: int | None = None,
) -> Path:
    if config.total_length != 81:
        raise ValueError(f"UniTraj-compatible SinD conversion expects 81 total steps, got {config.total_length}")
    records = discover_records(config.data_root, config.map_fallback_root, cities=cities)
    if max_records is not None:
        records = records[:max_records]
    windows: list[ScenarioWindow] = []
    for record in records:
        windows.extend(
            windows_for_record(
                record,
                past_len=config.past_len,
                future_len=config.future_len,
                stride=config.stride,
                max_scenarios=max_scenarios_per_record,
            )
        )
    if not windows:
        raise ValueError("No eligible SinD scenario windows were generated")
    output_dir = Path(config.canonical_scenario_root) / config.dataset_name
    _write_scenarios(windows, output_dir, dataset_name=config.dataset_name, dataset_version=config.dataset_version)
    return output_dir
