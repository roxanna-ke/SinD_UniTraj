from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from metadrive.type import MetaDriveType

from sind_converter.lights.bindings import LaneSignalBinding
from sind_converter.lights.channel_matching import RAW_FRAME_TO_TRACK_MS


RAW_TRAFFIC_LIGHT_STATE_MAP = {
    0: MetaDriveType.LANE_STATE_STOP,
    1: MetaDriveType.LANE_STATE_GO,
    2: MetaDriveType.LANE_STATE_CAUTION,
    3: MetaDriveType.LANE_STATE_CAUTION,
}


def _timestamp_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        normalized = col.lower().replace(" ", "").replace("_", "")
        if normalized in {"timestamp(ms)", "timestampms"}:
            return col
    return None


def _raw_frame_time_axis(df: pd.DataFrame) -> np.ndarray:
    if "RawFrameID" not in df.columns:
        ts_col = _timestamp_column(df)
        if ts_col is None:
            return np.asarray([], dtype=float)
        return pd.to_numeric(df[ts_col], errors="coerce").to_numpy(dtype=float)

    raw_frame = pd.to_numeric(df["RawFrameID"], errors="coerce").to_numpy(dtype=float)
    raw_time = raw_frame * RAW_FRAME_TO_TRACK_MS
    ts_col = _timestamp_column(df)
    if ts_col is None:
        return raw_time - np.nanmin(raw_time)

    timestamp = pd.to_numeric(df[ts_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(raw_frame) & np.isfinite(timestamp) & (timestamp > 0)
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(raw_frame[valid], timestamp[valid], 1)
        if np.isfinite(slope) and np.isfinite(intercept) and slope > 0:
            return raw_frame * float(slope) + float(intercept)
    return raw_time - np.nanmin(raw_time)


def dynamic_map_states_for_window(
    traffic_light: pd.DataFrame | None,
    lane_signal_bindings: tuple[LaneSignalBinding, ...],
    timestamps_ms: np.ndarray,
) -> dict[str, dict[str, Any]]:
    if traffic_light is None or traffic_light.empty or not lane_signal_bindings:
        return {}

    raw_times = _raw_frame_time_axis(traffic_light)
    if len(raw_times) == 0:
        return {}

    channel_state_lookup = _channel_state_lookup(traffic_light, raw_times)
    signal_states: dict[str, dict[str, Any]] = {}
    for binding in lane_signal_bindings:
        object_states = [
            _resolve_binding_state(channel_state_lookup, binding.traffic_light_channels, float(ts))
            for ts in timestamps_ms
        ]
        signal_id = f"signal_{binding.lane_feature_id}"
        signal_states[signal_id] = {
            "type": MetaDriveType.TRAFFIC_LIGHT,
            "lane": binding.lane_feature_id,
            "stop_point": np.asarray(binding.stop_point, dtype=np.float32),
            "state": {"object_state": object_states},
            "metadata": {
                "type": MetaDriveType.TRAFFIC_LIGHT,
                "object_id": signal_id,
                "stopline_id": binding.stopline_id,
                "movement": binding.movement,
                "raw_channels": list(binding.traffic_light_channels),
                "binding_group": binding.binding_group,
                "confidence": binding.confidence,
                "source": binding.source,
            },
        }
    return signal_states


def _channel_state_lookup(traffic_light: pd.DataFrame, raw_times: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for col in traffic_light.columns:
        if "traffic" not in str(col).lower() or "light" not in str(col).lower():
            continue
        raw_values = pd.to_numeric(traffic_light[col], errors="coerce").fillna(-1).astype(int).to_numpy()
        lookup[str(col)] = (raw_times, raw_values)
    return lookup


def _resolve_binding_state(
    channel_state_lookup: dict[str, tuple[np.ndarray, np.ndarray]],
    channels: tuple[str, ...],
    timestamp_ms: float,
) -> str:
    states = [_state_at_time(channel_state_lookup.get(channel), timestamp_ms) for channel in channels]
    states = [state for state in states if state is not None]
    if not states:
        return MetaDriveType.LANE_STATE_UNKNOWN
    counts = Counter(states)
    top_state, top_count = counts.most_common(1)[0]
    if list(counts.values()).count(top_count) > 1:
        return MetaDriveType.LANE_STATE_UNKNOWN
    return top_state


def _state_at_time(channel_state: tuple[np.ndarray, np.ndarray] | None, timestamp_ms: float) -> str | None:
    if channel_state is None:
        return None
    raw_times, raw_values = channel_state
    idx = int(np.searchsorted(raw_times, timestamp_ms, side="right") - 1)
    if idx < 0 or idx >= len(raw_values):
        return MetaDriveType.LANE_STATE_UNKNOWN
    return RAW_TRAFFIC_LIGHT_STATE_MAP.get(int(raw_values[idx]), MetaDriveType.LANE_STATE_UNKNOWN)
