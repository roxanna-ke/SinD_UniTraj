from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
SIND_DATASET = PROJECT_ROOT / "SinD" / "Dataset"
SIND_DATA = PROJECT_ROOT / "SinD" / "Data"
for path in [PROJECT_ROOT, PROJECT_ROOT / "scenarionet", REPO_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

scenarionet_common = pytest.importorskip("scenarionet.common_utils")
read_dataset_summary = scenarionet_common.read_dataset_summary
read_scenario = scenarionet_common.read_scenario


@pytest.mark.skipif(not SIND_DATASET.exists(), reason="SinD sample Dataset is not available")
def test_discover_records_handles_four_city_layouts():
    from sind_converter.data.discovery import discover_records

    records = discover_records(SIND_DATASET, SIND_DATA)
    cities = {record.city for record in records}
    assert {"Tianjin", "Changchun", "Chongqing", "Xi_an"}.issubset(cities)
    assert all(record.map_path.exists() for record in records)
    assert any(record.vehicle_meta_path is None for record in records)
    assert any(record.recording_meta_path and record.recording_meta_path.name == "recoding_metas.csv" for record in records)


@pytest.mark.skipif(not SIND_DATASET.exists(), reason="SinD sample Dataset is not available")
def test_modular_convert_writes_shared_multitarget_canonical_and_split(tmp_path):
    from sind_converter.config.defaults import ConverterConfig
    from sind_converter.scenarios.convert import convert_scenarios
    from sind_converter.splits.make import make_record_level_split

    cfg = ConverterConfig(
        data_root=SIND_DATASET,
        map_fallback_root=SIND_DATA,
        canonical_scenario_root=tmp_path / "canonical",
        split_root=tmp_path / "splits",
        cache_root=tmp_path / "cache",
        stride=80,
    )
    canonical_root = convert_scenarios(cfg, cities=["Tianjin"], max_records=2, max_scenarios_per_record=2)
    summary, scenario_files, mapping = read_dataset_summary(canonical_root)
    assert summary
    scenario = read_scenario(canonical_root, mapping, scenario_files[0])
    assert scenario["metadata"]["tracks_to_predict"]
    assert scenario["metadata"]["sdc_id"] == sorted(scenario["metadata"]["tracks_to_predict"].keys(), key=str)[0]
    assert {scenario["tracks"][tid]["metadata"]["type"] for tid in scenario["metadata"]["tracks_to_predict"]} <= {"car", "truck", "bus"}

    splits = make_record_level_split(canonical_root, cfg.split_root, seed=7)
    train_summary, _, _ = read_dataset_summary(splits["train"])
    test_summary, _, _ = read_dataset_summary(splits["test"])
    train_records = {meta["record_name"] for meta in train_summary.values()}
    test_records = {meta["record_name"] for meta in test_summary.values()}
    assert train_records.isdisjoint(test_records)


def test_agent_state_uses_frame_id_alignment_when_timestamps_differ():
    from sind_converter.scenarios.build import convert_window_to_scenario, generate_windows

    frames = np.arange(100, 181)
    target_rows = pd.DataFrame(
        {
            "track_id": ["target"] * len(frames),
            "frame_id": frames,
            "timestamp_ms": frames.astype(float) * 100.0 + 0.25,
            "agent_type": ["car"] * len(frames),
            "x": np.arange(len(frames), dtype=float),
            "y": np.zeros(len(frames), dtype=float),
            "vx": np.ones(len(frames), dtype=float),
            "vy": np.zeros(len(frames), dtype=float),
            "yaw_rad": np.zeros(len(frames), dtype=float),
            "length": np.full(len(frames), 4.5),
            "width": np.full(len(frames), 2.0),
        }
    )
    timestamp_source_rows = target_rows.copy()
    timestamp_source_rows["track_id"] = "timestamp_source"
    timestamp_source_rows["timestamp_ms"] = frames.astype(float) * 100.0
    vehicle_tracks = pd.concat([timestamp_source_rows, target_rows], ignore_index=True)
    pedestrian_tracks = pd.DataFrame(
        columns=["track_id", "frame_id", "timestamp_ms", "agent_type", "x", "y", "vx", "vy", "ax", "ay"]
    )

    windows = generate_windows(
        city="Changchun",
        record_name="synthetic",
        vehicle_tracks=vehicle_tracks,
        pedestrian_tracks=pedestrian_tracks,
        map_features={},
        lane_centers={},
        traffic_light=None,
        traffic_light_bindings=(),
        past_len=21,
        future_len=60,
        stride=81,
        max_scenarios=1,
    )

    scenario = convert_window_to_scenario(windows[0], dataset_version="v1")
    target_track = scenario["tracks"]["target"]
    assert target_track["state"]["valid"][20] == 1.0
    assert target_track["state"]["position"][20, 0] == 20.0
