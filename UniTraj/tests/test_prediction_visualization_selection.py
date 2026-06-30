from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
for path in [PROJECT_ROOT, PROJECT_ROOT / "scenarionet", REPO_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _record(name: str, displacement: float, *, is_target_track: bool = True) -> dict:
    pred_x = np.linspace(0.0, displacement, 60, dtype=np.float32)
    return {
        "scenario_id": f"scenario_{name}",
        "object_id": name,
        "past": np.stack([np.arange(21, dtype=np.float32), np.zeros(21, dtype=np.float32)], axis=-1),
        "gt": np.stack([np.arange(60, dtype=np.float32), np.ones(60, dtype=np.float32)], axis=-1),
        "pred": np.stack([pred_x, np.zeros_like(pred_x)], axis=-1),
        "top_probability": 0.5,
        "past_valid_count": 21,
        "gt_valid_count": 60,
        "pred_valid_count": 60,
        "is_target_track": is_target_track,
    }


def test_select_prediction_records_for_osm_map_uses_top4_target_predicted_displacement():
    from unitraj.utils.visualization import select_prediction_records_for_osm_map

    records = [
        _record("short", 1.0),
        _record("longest", 40.0),
        _record("non_target", 100.0, is_target_track=False),
        _record("third", 20.0),
        _record("second", 30.0),
        _record("fourth", 10.0),
        _record("extra", 5.0),
    ]

    selected = select_prediction_records_for_osm_map(records, max_tracks=4, min_total_steps=61)

    assert [record["object_id"] for record in selected] == ["longest", "second", "third", "fourth"]
    assert all(record["is_target_track"] for record in selected)
