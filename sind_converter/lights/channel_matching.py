from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sind_converter.data.discovery import RecordDescription


STATE_RED = 0
STATE_GREEN = 1
STATE_YELLOW = 3
STATE_UNKNOWN = -1

RAW_FRAME_TO_TRACK_MS = 100.0 / 3.0


@dataclass(frozen=True)
class MatchingConfig:
    min_support_crossings: int = 2
    high_score_threshold: float = 0.75
    medium_score_threshold: float = 0.55
    offset_search_ms: int = 20_000
    offset_step_ms: int = 500
    max_cycle_diff_s: float = 15.0


def normalize_channel_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).strip().lower())


def _timestamp_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        normalized = col.lower().replace(" ", "").replace("_", "")
        if normalized in {"timestamp(ms)", "timestampms"}:
            return col
    return None


def _state_name(state: int) -> str:
    return {
        STATE_RED: "red",
        STATE_GREEN: "green",
        STATE_YELLOW: "yellow",
        STATE_UNKNOWN: "unknown",
    }.get(int(state), "unknown")


def _channel_type(column: str) -> str:
    lower = column.lower()
    if "pedestrian" in lower:
        return "pedestrian"
    if "vehicle" in lower:
        return "vehicle"
    if "traffic" in lower and "light" in lower:
        return "vehicle"
    return "unknown"


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
    valid = np.isfinite(raw_time) & np.isfinite(timestamp) & (timestamp > 0)
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(raw_frame[valid], timestamp[valid], 1)
        if np.isfinite(slope) and np.isfinite(intercept) and slope > 0:
            return raw_frame * float(slope) + float(intercept)
    return raw_time - np.nanmin(raw_time)


def load_traffic_light_channels(record: RecordDescription) -> pd.DataFrame:
    if record.traffic_light_path is None or not Path(record.traffic_light_path).exists():
        return pd.DataFrame(columns=["timestamp_ms", "traffic_light_channel", "channel_type", "state"])
    df = pd.read_csv(record.traffic_light_path)
    event_times = _raw_frame_time_axis(df)
    if len(event_times) == 0:
        return pd.DataFrame(columns=["timestamp_ms", "traffic_light_channel", "channel_type", "state"])
    channel_columns = [col for col in df.columns if "traffic" in col.lower() and "light" in col.lower()]
    rows: list[dict[str, Any]] = []
    for col in channel_columns:
        channel_type = _channel_type(col)
        values = pd.to_numeric(df[col], errors="coerce").fillna(STATE_UNKNOWN).astype(int)
        raw_frames = pd.to_numeric(df["RawFrameID"], errors="coerce") if "RawFrameID" in df.columns else pd.Series([np.nan] * len(df))
        for raw_frame, ts, state in zip(raw_frames, event_times, values):
            if pd.isna(ts):
                continue
            rows.append(
                {
                    "raw_frame_id": float(raw_frame) if pd.notna(raw_frame) else "",
                    "traj_frame": float(ts) / 100.0,
                    "timestamp_ms": float(ts),
                    "traffic_light_channel": col,
                    "channel_type": channel_type,
                    "state": int(state),
                    "state_name": _state_name(int(state)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["timestamp_ms", "traffic_light_channel", "channel_type", "state"])
    df_channels = pd.DataFrame(rows).sort_values(["traffic_light_channel", "timestamp_ms", "raw_frame_id"])
    return df_channels.drop_duplicates(["traffic_light_channel", "timestamp_ms"], keep="last")


def phase_event_rows(record: RecordDescription) -> list[dict[str, Any]]:
    channels = load_traffic_light_channels(record)
    if channels.empty:
        return []
    rows: list[dict[str, Any]] = []
    for row in channels.to_dict("records"):
        rows.append(
            {
                "city": record.city,
                "record_name": record.record_name,
                "traffic_light_channel": row["traffic_light_channel"],
                "channel_type": row["channel_type"],
                "raw_frame_id": row.get("raw_frame_id", ""),
                "traj_frame": row.get("traj_frame", ""),
                "time_ms": row["timestamp_ms"],
                "state": row["state"],
                "state_name": row["state_name"],
            }
        )
    return rows


def infer_channel_cycles(record: RecordDescription) -> list[dict[str, Any]]:
    channels = load_traffic_light_channels(record)
    if channels.empty:
        return []
    rows: list[dict[str, Any]] = []
    for channel, channel_df in channels.groupby("traffic_light_channel", sort=True):
        channel_df = channel_df.sort_values("timestamp_ms").copy()
        times = channel_df["timestamp_ms"].to_numpy(dtype=float)
        states = channel_df["state"].to_numpy(dtype=int)
        next_times = np.roll(times, -1)
        durations_s = (next_times[:-1] - times[:-1]) / 1000.0
        state_durations: dict[int, list[float]] = {STATE_RED: [], STATE_GREEN: [], STATE_YELLOW: []}
        for state, duration in zip(states[:-1], durations_s):
            if duration > 0 and int(state) in state_durations:
                state_durations[int(state)].append(float(duration))

        green_starts = times[states == STATE_GREEN] / 1000.0
        cycle_intervals = np.diff(green_starts)
        valid_cycles = cycle_intervals[(cycle_intervals >= 30.0) & (cycle_intervals <= 240.0)]
        estimated_cycle_s = float(np.median(valid_cycles)) if len(valid_cycles) else ""
        confidence = "high" if len(valid_cycles) >= 3 else "medium" if len(valid_cycles) >= 1 else "low"
        rows.append(
            {
                "city": record.city,
                "record_name": record.record_name,
                "traffic_light_channel": channel,
                "channel_type": str(channel_df["channel_type"].iloc[0]),
                "estimated_cycle_s": estimated_cycle_s,
                "red_duration_s": _median_or_empty(state_durations[STATE_RED]),
                "green_duration_s": _median_or_empty(state_durations[STATE_GREEN]),
                "yellow_duration_s": _median_or_empty(state_durations[STATE_YELLOW]),
                "phase_offset_ms": float(times[0]) if len(times) else "",
                "num_cycles": int(len(valid_cycles)),
                "confidence": confidence,
            }
        )
    return rows


def _median_or_empty(values: list[float]) -> float | str:
    return float(np.median(values)) if values else ""


def _state_at_times(channel_df: pd.DataFrame, times_ms: np.ndarray) -> np.ndarray:
    if channel_df.empty or len(times_ms) == 0:
        return np.asarray([], dtype=int)
    channel_df = channel_df.sort_values("timestamp_ms")
    raw_times = channel_df["timestamp_ms"].to_numpy(dtype=float)
    raw_states = channel_df["state"].to_numpy(dtype=int)
    indices = np.searchsorted(raw_times, times_ms, side="right") - 1
    states = np.full(len(times_ms), STATE_UNKNOWN, dtype=int)
    valid = indices >= 0
    states[valid] = raw_states[indices[valid]]
    return states


def _green_intervals(channel_df: pd.DataFrame) -> list[tuple[float, float]]:
    if channel_df.empty:
        return []
    channel_df = channel_df.sort_values("timestamp_ms")
    times = channel_df["timestamp_ms"].to_numpy(dtype=float)
    states = channel_df["state"].to_numpy(dtype=int)
    intervals: list[tuple[float, float]] = []
    for idx, state in enumerate(states[:-1]):
        start = float(times[idx])
        end = float(times[idx + 1])
        if int(state) == STATE_GREEN and end > start:
            intervals.append((start, end))
    return intervals


def _share_in_intervals(times_ms: np.ndarray, intervals: list[tuple[float, float]]) -> float:
    if len(times_ms) == 0 or not intervals:
        return 0.0
    hits = np.zeros(len(times_ms), dtype=bool)
    for start, end in intervals:
        hits |= (times_ms >= start) & (times_ms < end)
    return float(np.mean(hits))


def _offset_search(times_ms: np.ndarray, green_intervals: list[tuple[float, float]], cfg: MatchingConfig) -> tuple[float, float, float]:
    base_score = _share_in_intervals(times_ms, green_intervals)
    best_offset = 0.0
    best_score = base_score
    for offset in range(-cfg.offset_search_ms, cfg.offset_search_ms + 1, cfg.offset_step_ms):
        score = _share_in_intervals(times_ms + float(offset), green_intervals)
        if score > best_score:
            best_score = score
            best_offset = float(offset)
    return best_offset, base_score, best_score


def _cycle_lookup(cycle_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], float]:
    lookup: dict[tuple[str, str, str, str], float] = {}
    for row in cycle_rows:
        value = row.get("estimated_cycle_s", "")
        if value == "" or pd.isna(value):
            continue
        lookup[(row["city"], row["record_name"], str(row["stopline_id"]), row["movement"])] = float(value)
    return lookup


def _channel_cycle_lookup(channel_cycle_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], float]:
    lookup: dict[tuple[str, str, str], float] = {}
    for row in channel_cycle_rows:
        value = row.get("estimated_cycle_s", "")
        if value == "" or pd.isna(value):
            continue
        lookup[(row["city"], row["record_name"], row["traffic_light_channel"])] = float(value)
    return lookup


def _relation_lookup(relation_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in relation_rows:
        stopline = row.get("osm_relation_expected_stopline", "")
        if stopline and stopline not in lookup:
            lookup[stopline] = row
    return lookup


def _relation_fields(
    city_has_relations: bool,
    relation_by_stopline: dict[str, dict[str, str]],
    stopline_id: str,
    channel: str,
) -> tuple[str, str, str, str]:
    if not city_has_relations:
        return "none", "none", "none", "not_applicable"
    relation = relation_by_stopline.get(stopline_id)
    if relation is None:
        return "none", "none", "none", "false"
    expected_channel = relation.get("osm_relation_expected_channel", "unknown") or "unknown"
    expected_stopline = relation.get("osm_relation_expected_stopline", "unknown") or "unknown"
    relation_id = relation.get("osm_relation_id", "unknown") or "unknown"
    if expected_channel == "unknown":
        return expected_channel, expected_stopline, relation_id, "stopline_only"
    agrees = normalize_channel_name(expected_channel) == normalize_channel_name(channel)
    return expected_channel, expected_stopline, relation_id, "true" if agrees else "false"


def _confidence(score: float, support: int, relation_agrees: str, cfg: MatchingConfig) -> str:
    if relation_agrees == "true" and support >= 1 and score >= cfg.medium_score_threshold:
        return "high"
    if relation_agrees == "true" and support >= 1:
        return "medium"
    if support >= cfg.min_support_crossings and score >= cfg.high_score_threshold:
        return "high"
    if support >= 1 and score >= cfg.medium_score_threshold:
        return "medium"
    return "low"


def _source(confidence: str, relation_agrees: str) -> str:
    if confidence == "low":
        return "manual_review"
    if relation_agrees == "true":
        return "trajectory_csv_match+osm_relation_validated"
    if relation_agrees == "false":
        return "trajectory_csv_match_osm_conflict"
    if relation_agrees == "stopline_only":
        return "trajectory_csv_match+osm_relation_stopline_prior"
    return "trajectory_csv_match"


def match_record_channels(
    record: RecordDescription,
    crossing_rows: list[dict[str, Any]],
    window_rows: list[dict[str, Any]],
    cycle_rows: list[dict[str, Any]],
    channel_cycle_rows: list[dict[str, Any]],
    relation_rows: list[dict[str, str]],
    cfg: MatchingConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    channels = load_traffic_light_channels(record)
    if channels.empty:
        return [], []

    crossings = pd.DataFrame(crossing_rows)
    city_relations = [row for row in relation_rows if row.get("city") == record.city]
    city_has_relations = bool(city_relations)
    relation_by_stopline = _relation_lookup(city_relations)
    crossing_cycles = _cycle_lookup(cycle_rows)
    channel_cycles = _channel_cycle_lookup(channel_cycle_rows)
    audit_rows: list[dict[str, Any]] = []

    movement_groups: list[tuple[tuple[str, str], pd.DataFrame]] = []
    if not crossings.empty:
        movement_groups = list(crossings.groupby(["stopline_id", "movement"], sort=True))

    for channel, channel_df in channels.groupby("traffic_light_channel", sort=True):
        channel_type = str(channel_df["channel_type"].iloc[0])
        if not movement_groups or channel_type != "vehicle":
            audit_rows.append(
                _unmatched_audit_row(record, str(channel), channel_type, "no vehicle movement candidates" if not movement_groups else "non-vehicle channel")
            )
            continue
        for (stopline_id, movement), movement_crossings in movement_groups:
            times = movement_crossings["crossing_timestamp_ms"].to_numpy(dtype=float)
            states = _state_at_times(channel_df, times)
            if len(states) == 0:
                continue
            intervals = _green_intervals(channel_df)
            best_offset_ms, score_before_offset, score_after_offset = _offset_search(times, intervals, cfg)
            crossing_green_share = score_after_offset
            csv_cycle_s = channel_cycles.get((record.city, record.record_name, str(channel)), "")
            crossing_cycle_s = crossing_cycles.get((record.city, record.record_name, str(stopline_id), str(movement)), "")
            cycle_diff_s = ""
            cycle_score = 0.5
            if csv_cycle_s != "" and crossing_cycle_s != "":
                cycle_diff_s = abs(float(csv_cycle_s) - float(crossing_cycle_s))
                cycle_score = max(0.0, 1.0 - float(cycle_diff_s) / cfg.max_cycle_diff_s)
            score = 0.9 * crossing_green_share + 0.1 * cycle_score
            expected_channel, expected_stopline, relation_id, relation_agrees = _relation_fields(
                city_has_relations,
                relation_by_stopline,
                str(stopline_id),
                str(channel),
            )
            support = int(len(states))
            confidence = _confidence(score, support, relation_agrees, cfg)
            source = _source(confidence, relation_agrees)
            audit_rows.append(
                {
                    "city": record.city,
                    "record_name": record.record_name,
                    "traffic_light_channel": channel,
                    "channel_type": channel_type,
                    "stopline_id": stopline_id,
                    "osm_way_id": str(stopline_id),
                    "movement": movement,
                    "trajectory_match_score": score,
                    "trajectory_rank": "",
                    "num_support_crossings": support,
                    "csv_cycle_s": csv_cycle_s,
                    "crossing_cycle_s": crossing_cycle_s,
                    "cycle_diff_s": cycle_diff_s,
                    "best_offset_ms": best_offset_ms,
                    "score_before_offset": score_before_offset,
                    "score_after_offset": score_after_offset,
                    "osm_relation_expected_channel": expected_channel,
                    "osm_relation_expected_stopline": expected_stopline,
                    "osm_relation_id": relation_id,
                    "osm_relation_agrees": relation_agrees,
                    "final_confidence": confidence,
                    "final_source": source,
                    "notes": f"crossing_green_share_after_offset={crossing_green_share:.3f};cycle_score={cycle_score:.3f}",
                }
            )

    _assign_ranks(audit_rows)
    final_rows = [_final_row(row) for row in audit_rows if _use_as_final_mapping(row, cfg)]
    return audit_rows, final_rows


def _use_as_final_mapping(row: dict[str, Any], cfg: MatchingConfig) -> bool:
    if row.get("final_confidence") not in {"high", "medium"}:
        return False
    if row.get("osm_relation_agrees") == "true":
        return True
    cycle_diff = row.get("cycle_diff_s", "")
    if cycle_diff != "" and not pd.isna(cycle_diff) and float(cycle_diff) > cfg.max_cycle_diff_s:
        return False
    return row.get("trajectory_rank") == 1


def _unmatched_audit_row(record: RecordDescription, channel: str, channel_type: str, notes: str) -> dict[str, Any]:
    return {
        "city": record.city,
        "record_name": record.record_name,
        "traffic_light_channel": channel,
        "channel_type": channel_type,
        "stopline_id": "unmatched",
        "osm_way_id": "unmatched",
        "movement": "unmatched",
        "trajectory_match_score": 0.0,
        "trajectory_rank": "",
        "num_support_crossings": 0,
        "csv_cycle_s": "",
        "crossing_cycle_s": "",
        "cycle_diff_s": "",
        "best_offset_ms": "",
        "score_before_offset": "",
        "score_after_offset": "",
        "osm_relation_expected_channel": "none",
        "osm_relation_expected_stopline": "none",
        "osm_relation_id": "none",
        "osm_relation_agrees": "not_applicable",
        "final_confidence": "unmatched",
        "final_source": "unmatched",
        "notes": notes,
    }


def _assign_ranks(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("stopline_id") == "unmatched":
            continue
        key = (row["city"], row["record_name"], row["stopline_id"], row["movement"])
        grouped.setdefault(key, []).append(row)
    for group in grouped.values():
        ranked = sorted(group, key=lambda row: float(row["trajectory_match_score"]), reverse=True)
        for idx, row in enumerate(ranked, start=1):
            row["trajectory_rank"] = idx


def _final_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "city": row["city"],
        "record_name": row["record_name"],
        "traffic_light_channel": row["traffic_light_channel"],
        "channel_type": row["channel_type"],
        "stopline_id": row["stopline_id"],
        "osm_way_id": row["osm_way_id"],
        "movement": row["movement"],
        "match_score": row["trajectory_match_score"],
        "confidence": row["final_confidence"],
        "num_support_crossings": row["num_support_crossings"],
        "csv_cycle_s": row["csv_cycle_s"],
        "crossing_cycle_s": row["crossing_cycle_s"],
        "cycle_diff_s": row["cycle_diff_s"],
        "best_offset_ms": row["best_offset_ms"],
        "score_before_offset": row["score_before_offset"],
        "score_after_offset": row["score_after_offset"],
        "source": row["final_source"],
        "notes": row["notes"],
    }


AUDIT_FIELDS = [
    "city",
    "record_name",
    "traffic_light_channel",
    "channel_type",
    "stopline_id",
    "osm_way_id",
    "movement",
    "trajectory_match_score",
    "trajectory_rank",
    "num_support_crossings",
    "csv_cycle_s",
    "crossing_cycle_s",
    "cycle_diff_s",
    "best_offset_ms",
    "score_before_offset",
    "score_after_offset",
    "osm_relation_expected_channel",
    "osm_relation_expected_stopline",
    "osm_relation_id",
    "osm_relation_agrees",
    "final_confidence",
    "final_source",
    "notes",
]

FINAL_MAPPING_FIELDS = [
    "city",
    "record_name",
    "traffic_light_channel",
    "channel_type",
    "stopline_id",
    "osm_way_id",
    "movement",
    "match_score",
    "confidence",
    "num_support_crossings",
    "csv_cycle_s",
    "crossing_cycle_s",
    "cycle_diff_s",
    "best_offset_ms",
    "score_before_offset",
    "score_after_offset",
    "source",
    "notes",
]

PHASE_EVENT_FIELDS = [
    "city",
    "record_name",
    "traffic_light_channel",
    "channel_type",
    "raw_frame_id",
    "traj_frame",
    "time_ms",
    "state",
    "state_name",
]

CHANNEL_CYCLE_FIELDS = [
    "city",
    "record_name",
    "traffic_light_channel",
    "channel_type",
    "estimated_cycle_s",
    "red_duration_s",
    "green_duration_s",
    "yellow_duration_s",
    "phase_offset_ms",
    "num_cycles",
    "confidence",
]
