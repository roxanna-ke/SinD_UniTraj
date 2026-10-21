from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
for path in [PROJECT_ROOT, PROJECT_ROOT / "scenarionet", REPO_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _record(
    name: str,
    displacement: float,
    *,
    gt_displacement: float | None = None,
    y_offset: float = 0.0,
    is_target_track: bool = True,
    object_type: int = 1,
) -> dict:
    if gt_displacement is None:
        gt_displacement = displacement
    pred_x = np.linspace(0.0, displacement, 60, dtype=np.float32)
    gt_x = np.linspace(0.0, gt_displacement, 60, dtype=np.float32)
    return {
        "scenario_id": f"scenario_{name}",
        "object_id": name,
        "past": np.stack([np.linspace(-20.0, 0.0, 21, dtype=np.float32), np.full(21, y_offset, dtype=np.float32)], axis=-1),
        "gt": np.stack([gt_x, np.full_like(gt_x, y_offset)], axis=-1),
        "pred": np.stack([pred_x, np.full_like(pred_x, y_offset)], axis=-1),
        "object_type": object_type,
        "top_probability": 0.5,
        "past_valid_count": 21,
        "gt_valid_count": 60,
        "pred_valid_count": 60,
        "is_target_track": is_target_track,
    }


def test_select_prediction_records_for_osm_map_uses_moving_nonoverlapping_target_vehicles():
    from unitraj.utils.visualization import select_prediction_records_for_osm_map

    records = [
        _record("static_pred", 1.0, y_offset=0.0),
        _record("longest", 40.0, y_offset=0.0),
        _record("non_target", 100.0, y_offset=20.0, is_target_track=False),
        _record("pedestrian_target", 90.0, y_offset=30.0, object_type=2),
        _record("third", 20.0, y_offset=40.0),
        _record("second", 30.0, y_offset=60.0),
        _record("fourth", 10.0, y_offset=80.0),
        _record("extra", 5.0, y_offset=100.0),
    ]

    selected = select_prediction_records_for_osm_map(
        records,
        max_tracks=4,
        min_track_distance=4.0,
        min_total_steps=61,
        min_displacement=2.0,
    )

    assert [record["object_id"] for record in selected] == ["longest", "second", "third", "fourth"]
    assert all(record["is_target_track"] for record in selected)
    assert all(record["object_type"] == 1 for record in selected)


def test_select_prediction_records_for_osm_map_filters_overlapping_tracks():
    from unitraj.utils.visualization import select_prediction_records_for_osm_map

    records = [
        _record("best_overlap", 40.0, y_offset=0.0),
        _record("worse_overlap", 35.0, y_offset=1.0),
        _record("separate", 30.0, y_offset=20.0),
    ]

    selected = select_prediction_records_for_osm_map(
        records,
        max_tracks=3,
        min_track_distance=4.0,
        min_total_steps=61,
        min_displacement=2.0,
    )

    assert [record["object_id"] for record in selected] == ["best_overlap", "separate"]
