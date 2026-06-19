from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
for path in [PROJECT_ROOT, PROJECT_ROOT / "scenarionet", REPO_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def test_resolve_signal_bindings_and_emit_dynamic_map_states(tmp_path):
    from sind_converter.data.discovery import RecordDescription
    from sind_converter.lights.bindings import resolve_signal_bindings
    from sind_converter.lights.standardize import dynamic_map_states_for_window
    from sind_converter.lights.stopline_extraction import StopLine

    binding_root = tmp_path / "bindings"
    (binding_root / "channel_groups").mkdir(parents=True)
    (binding_root / "lane_bindings").mkdir(parents=True)
    (binding_root / "channel_groups" / "Xi_an.yaml").write_text(
        '{"city":"Xi_an","channel_groups":[{"group_id":"xi_phase_a","traffic_light_channels":["Traffic light 1"],"movements":[{"stopline_id":"10","movement":"straight"}]}]}',
        encoding="utf-8",
    )
    (binding_root / "lane_bindings" / "Xi_an.yaml").write_text(
        '{"city":"Xi_an","lane_bindings":[{"stopline_id":"10","movement":"straight","lane_ids":["lane_20"]}]}',
        encoding="utf-8",
    )

    record = RecordDescription(
        city="Xi_an",
        record_name="record_001",
        record_dir=tmp_path,
        vehicle_tracks_path=tmp_path / "veh.csv",
        pedestrian_tracks_path=tmp_path / "ped.csv",
        traffic_light_path=tmp_path / "lights.csv",
        vehicle_meta_path=None,
        pedestrian_meta_path=None,
        recording_meta_path=None,
        map_path=tmp_path / "map.osm",
    )
    traffic_light = pd.DataFrame(
        {
            "timestamp(ms)": [0.0, 1000.0, 2000.0],
            "Traffic light 1": [0, 1, 3],
        }
    )
    map_features = {
        "lane_20": {
            "type": "LANE_SURFACE_STREET",
            "polyline": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
        }
    }
    stop_lines = [
        StopLine(
            city="Xi_an",
            stopline_id="10",
            osm_way_id="10",
            x1=1.0,
            y1=-1.0,
            x2=1.0,
            y2=1.0,
            mid_x=1.0,
            mid_y=0.0,
            nx=1.0,
            ny=0.0,
        )
    ]

    bindings = resolve_signal_bindings(record, traffic_light, map_features, stop_lines, binding_root)
    assert len(bindings) == 1
    assert bindings[0].lane_feature_id == "lane_20"
    assert bindings[0].stopline_id == "10"
    assert bindings[0].movement == "straight"

    dynamic_map_states = dynamic_map_states_for_window(
        traffic_light=traffic_light,
        lane_signal_bindings=bindings,
        timestamps_ms=np.array([0.0, 1000.0, 2000.0], dtype=np.float32),
    )
    assert set(dynamic_map_states) == {"signal_lane_20"}
    signal = dynamic_map_states["signal_lane_20"]
    assert signal["lane"] == "lane_20"
    assert signal["metadata"]["stopline_id"] == "10"
    assert signal["state"]["object_state"] == ["LANE_STATE_STOP", "LANE_STATE_GO", "LANE_STATE_CAUTION"]


def test_base_dataset_encodes_center_frame_lane_control_state_into_map_tokens():
    from unitraj.datasets.base_dataset import BaseDataset

    dataset = BaseDataset.__new__(BaseDataset)
    dataset.config = {
        "use_lane_control_state_in_map_tokens": True,
        "max_num_roads": 8,
        "max_points_per_lane": 20,
        "line_type": ["lane"],
        "map_range": 100.0,
        "center_offset_of_map": (30.0, 0.0),
        "point_sampled_interval": 1,
        "vector_break_dist_thresh": 1.0,
        "num_points_each_polyline": 20,
        "method": {"model_name": "MTR"},
    }
    map_infos = {
        "lane": [
            {
                "id": "lane_20",
                "polyline_index": (0, 3),
                "light_state": 4,
            }
        ],
        "road_line": [],
        "road_edge": [],
        "stop_sign": [],
        "crosswalk": [],
        "speed_bump": [],
        "all_polylines": np.array(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 2.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 2.0],
                [2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 2.0],
            ],
            dtype=np.float32,
        ),
    }
    center_objects = np.array([[0.0, 0.0, 0.0, 4.5, 2.0, 1.5, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    map_polylines, map_mask, _ = dataset.get_manually_split_map_data(center_objects, map_infos)
    assert map_polylines.shape[-1] == 38
    assert map_mask.any()
    first_valid = map_polylines[0, 0, 0]
    light_one_hot = first_valid[9:18]
    assert light_one_hot.shape[0] == 9
    assert light_one_hot[4] == 1.0
    assert light_one_hot.sum() == 1.0
