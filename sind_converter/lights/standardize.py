from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from metadrive.type import MetaDriveType


TRAFFIC_LIGHT_STATE_MAP = {
    0: MetaDriveType.LANE_STATE_UNKNOWN,
    1: MetaDriveType.LANE_STATE_STOP,
    2: MetaDriveType.LANE_STATE_CAUTION,
    3: MetaDriveType.LANE_STATE_GO,
}


def _timestamp_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if col.lower().replace(" ", "") in {"timestamp(ms)", "timestamp_ms"}:
            return col
    return None


def dynamic_map_states_for_window(
    traffic_light: pd.DataFrame | None,
    lane_centers: dict[str, np.ndarray],
    timestamps_ms: np.ndarray,
) -> dict[str, dict[str, Any]]:
    if traffic_light is None or traffic_light.empty or not lane_centers:
        return {}
    ts_col = _timestamp_column(traffic_light)
    if ts_col is None:
        return {}
    signal_columns = [col for col in traffic_light.columns if "traffic light" in col.lower()]
    if not signal_columns:
        return {}

    lane_names = list(lane_centers.keys())
    traffic_df = traffic_light.sort_values(ts_col)
    raw_times = traffic_df[ts_col].to_numpy(dtype=float)
    signal_states: dict[str, dict[str, Any]] = {}
    for idx, col in enumerate(signal_columns):
        lane_name = lane_names[idx % len(lane_names)]
        lane_center = lane_centers[lane_name]
        stop_point = lane_center[len(lane_center) // 2] if len(lane_center) else np.zeros(3, dtype=np.float32)
        raw_values = traffic_df[col].fillna(0).to_numpy(dtype=int)
        object_states = []
        for ts in timestamps_ms:
            state_idx = np.searchsorted(raw_times, ts, side="right") - 1
            raw_state = int(raw_values[state_idx]) if state_idx >= 0 else 0
            object_states.append(TRAFFIC_LIGHT_STATE_MAP.get(raw_state, MetaDriveType.LANE_STATE_UNKNOWN))
        signal_id = f"signal_{idx + 1}"
        signal_states[signal_id] = {
            "type": MetaDriveType.TRAFFIC_LIGHT,
            "lane": f"lane_{lane_name}",
            "stop_point": stop_point.astype(np.float32),
            "state": {"object_state": object_states},
            "metadata": {"raw_column": col, "type": MetaDriveType.TRAFFIC_LIGHT, "object_id": signal_id},
        }
    return signal_states
