from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sind_converter.data.discovery import RecordDescription
from sind_converter.lights.stopline_extraction import StopLine


VEHICLE_TARGET_TYPES = {"car", "truck", "bus"}


@dataclass(frozen=True)
class InferenceConfig:
    segment_tolerance_m: float = 2.0
    min_crossing_speed_mps: float = 0.5
    dedupe_window_ms: float = 1000.0
    heading_turn_threshold_deg: float = 35.0
    min_track_frames: int = 15
    min_track_duration_ms: float = 1500.0
    max_frame_gap_ms: float = 300.0
    min_crossing_context_frames: int = 5
    min_pre_crossing_displacement_m: float = 3.0
    min_post_crossing_displacement_m: float = 3.0
    burst_gap_s: float = 4.0
    min_cycle_s: float = 60.0
    max_cycle_s: float = 240.0


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _finite_heading(row: pd.Series) -> float:
    for col in ("yaw_rad", "heading_rad"):
        if col in row and pd.notna(row[col]) and np.isfinite(float(row[col])):
            return float(row[col])
    vx = float(row.get("vx", 0.0) or 0.0)
    vy = float(row.get("vy", 0.0) or 0.0)
    if abs(vx) > 1e-6 or abs(vy) > 1e-6:
        return float(math.atan2(vy, vx))
    return 0.0


def _movement_from_delta(delta_heading: float, threshold_deg: float) -> str:
    threshold = math.radians(threshold_deg)
    if abs(delta_heading) < threshold:
        return "straight"
    if delta_heading >= threshold:
        return "left"
    if delta_heading <= -threshold:
        return "right"
    return "unknown"


def _point_segment_distance(point: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[float, float]:
    segment = p2 - p1
    length_sq = float(np.dot(segment, segment))
    if length_sq == 0.0:
        return float(np.linalg.norm(point - p1)), 0.0
    projection = float(np.dot(point - p1, segment) / length_sq)
    clamped = min(1.0, max(0.0, projection))
    closest = p1 + clamped * segment
    return float(np.linalg.norm(point - closest)), projection


def _load_vehicle_tracks(record: RecordDescription) -> pd.DataFrame:
    df = pd.read_csv(record.vehicle_tracks_path)
    required = {"track_id", "frame_id", "timestamp_ms", "agent_type", "x", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{record.vehicle_tracks_path} is missing required columns: {sorted(missing)}")
    agent_type = df["agent_type"].astype(str).str.lower()
    df = df[agent_type.isin(VEHICLE_TARGET_TYPES)].copy()
    if df.empty:
        return df
    numeric_cols = ["frame_id", "timestamp_ms", "x", "y", "vx", "vy", "yaw_rad", "heading_rad"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    finite_mask = np.isfinite(df[["timestamp_ms", "x", "y"]].to_numpy(dtype=np.float64)).all(axis=1)
    df = df[finite_mask].copy()
    df["track_id"] = df["track_id"].astype(str)
    df = df.sort_values(["track_id", "frame_id", "timestamp_ms"])
    return df


def _filter_track_quality(tracks: pd.DataFrame, cfg: InferenceConfig) -> pd.DataFrame:
    if tracks.empty:
        return tracks
    kept_tracks: list[pd.DataFrame] = []
    for _, track in tracks.groupby("track_id", sort=True):
        if len(track) < cfg.min_track_frames:
            continue
        timestamps = track["timestamp_ms"].to_numpy(dtype=np.float64)
        duration_ms = float(timestamps[-1] - timestamps[0])
        if duration_ms < cfg.min_track_duration_ms:
            continue
        if len(timestamps) >= 2:
            frame_gaps = np.diff(timestamps)
            if len(frame_gaps) and float(np.max(frame_gaps)) > cfg.max_frame_gap_ms:
                continue
        kept_tracks.append(track)
    if not kept_tracks:
        return tracks.iloc[0:0].copy()
    return pd.concat(kept_tracks, ignore_index=True)


def _cumulative_displacement(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    steps = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(steps, axis=1)))


def _crossings_for_track(
    record: RecordDescription,
    track: pd.DataFrame,
    stop_lines: list[StopLine],
    cfg: InferenceConfig,
) -> list[dict[str, Any]]:
    if len(track) < 2:
        return []
    points = track[["x", "y"]].to_numpy(dtype=np.float64)
    timestamps = track["timestamp_ms"].to_numpy(dtype=np.float64)
    frames = track["frame_id"].to_numpy()
    vx = track["vx"].to_numpy(dtype=np.float64) if "vx" in track.columns else np.zeros(len(track), dtype=np.float64)
    vy = track["vy"].to_numpy(dtype=np.float64) if "vy" in track.columns else np.zeros(len(track), dtype=np.float64)
    track_id = str(track["track_id"].iloc[0])
    track_series = [track.iloc[i] for i in range(len(track))]
    candidate_rows: list[dict[str, Any]] = []

    for stopline in stop_lines:
        centered = points - stopline.midpoint
        distances = centered[:, 0] * stopline.nx + centered[:, 1] * stopline.ny
        for idx in range(len(track) - 1):
            if idx < cfg.min_crossing_context_frames - 1:
                continue
            if len(track) - (idx + 1) < cfg.min_crossing_context_frames:
                continue
            d0 = float(distances[idx])
            d1 = float(distances[idx + 1])
            if d0 == 0.0 and d1 == 0.0:
                continue
            if d0 * d1 > 0.0:
                continue
            denom = abs(d0) + abs(d1)
            alpha = 0.0 if denom == 0.0 else abs(d0) / denom
            crossing_point = points[idx] + alpha * (points[idx + 1] - points[idx])
            segment_distance, projection = _point_segment_distance(crossing_point, stopline.p1, stopline.p2)
            segment = stopline.p2 - stopline.p1
            segment_len = max(float(np.linalg.norm(segment)), 1e-6)
            projection_tol = cfg.segment_tolerance_m / segment_len
            if segment_distance > cfg.segment_tolerance_m or projection < -projection_tol or projection > 1.0 + projection_tol:
                continue
            speed = float((math.hypot(vx[idx], vy[idx]) + math.hypot(vx[idx + 1], vy[idx + 1])) / 2.0)
            if speed < cfg.min_crossing_speed_mps:
                continue
            pre_points = points[idx - cfg.min_crossing_context_frames + 1 : idx + 1]
            post_points = points[idx + 1 : idx + 1 + cfg.min_crossing_context_frames]
            if _cumulative_displacement(pre_points) < cfg.min_pre_crossing_displacement_m:
                continue
            if _cumulative_displacement(post_points) < cfg.min_post_crossing_displacement_m:
                continue
            crossing_ts = float(timestamps[idx] + alpha * (timestamps[idx + 1] - timestamps[idx]))
            heading_before = _finite_heading(track_series[idx])
            final_heading = _finite_heading(track_series[-1])
            delta_heading = wrap_to_pi(final_heading - heading_before)
            movement = _movement_from_delta(delta_heading, cfg.heading_turn_threshold_deg)
            if movement not in {"left", "straight"}:
                continue
            candidate_rows.append(
                {
                    "city": record.city,
                    "record_name": record.record_name,
                    "track_id": track_id,
                    "vehicle_global_id": f"{record.city}/{record.record_name}/{track_id}",
                    "stopline_id": stopline.stopline_id,
                    "osm_way_id": stopline.osm_way_id,
                    "crossing_frame_before": frames[idx],
                    "crossing_timestamp_ms": crossing_ts,
                    "crossing_x": float(crossing_point[0]),
                    "crossing_y": float(crossing_point[1]),
                    "speed": speed,
                    "heading_before": heading_before,
                    "heading_final": final_heading,
                    "delta_heading": delta_heading,
                    "movement": movement,
                    "_segment_distance": segment_distance,
                }
            )
    return _dedupe_track_crossings(candidate_rows, cfg.dedupe_window_ms)


def _dedupe_track_crossings(rows: list[dict[str, Any]], window_ms: float) -> list[dict[str, Any]]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda row: float(row["crossing_timestamp_ms"]))
    kept: list[dict[str, Any]] = []
    cluster: list[dict[str, Any]] = [rows[0]]
    for row in rows[1:]:
        if float(row["crossing_timestamp_ms"]) - float(cluster[-1]["crossing_timestamp_ms"]) <= window_ms:
            cluster.append(row)
            continue
        kept.append(min(cluster, key=lambda item: float(item["_segment_distance"])))
        cluster = [row]
    kept.append(min(cluster, key=lambda item: float(item["_segment_distance"])))
    for row in kept:
        row.pop("_segment_distance", None)
    return kept


def infer_crossing_events(
    record: RecordDescription,
    stop_lines: list[StopLine],
    cfg: InferenceConfig,
) -> list[dict[str, Any]]:
    tracks = _load_vehicle_tracks(record)
    tracks = _filter_track_quality(tracks, cfg)
    if tracks.empty:
        return []
    rows: list[dict[str, Any]] = []
    for _, track in tracks.groupby("track_id", sort=True):
        rows.extend(_crossings_for_track(record, track, stop_lines, cfg))
    return rows


def infer_movement_bursts(crossing_rows: list[dict[str, Any]], cfg: InferenceConfig) -> list[dict[str, Any]]:
    if not crossing_rows:
        return []
    df = pd.DataFrame(crossing_rows)
    group_cols = ["city", "record_name", "stopline_id", "movement"]
    burst_rows: list[dict[str, Any]] = []
    gap_ms = cfg.burst_gap_s * 1000.0
    for key, group in df.sort_values("crossing_timestamp_ms").groupby(group_cols, sort=True):
        city, record_name, stopline_id, movement = key
        current: list[float] = []
        burst_idx = 0
        for ts in group["crossing_timestamp_ms"].astype(float):
            if current and ts - current[-1] > gap_ms:
                burst_rows.append(_burst_row(city, record_name, stopline_id, movement, burst_idx, current))
                burst_idx += 1
                current = []
            current.append(float(ts))
        if current:
            burst_rows.append(_burst_row(city, record_name, stopline_id, movement, burst_idx, current))
    return burst_rows


def _burst_row(city: str, record_name: str, stopline_id: str, movement: str, burst_idx: int, times: list[float]) -> dict[str, Any]:
    headways = np.diff(times) / 1000.0 if len(times) > 1 else np.asarray([], dtype=float)
    return {
        "city": city,
        "record_name": record_name,
        "stopline_id": stopline_id,
        "movement": movement,
        "burst_id": f"{city}/{record_name}/{stopline_id}/{movement}/{burst_idx}",
        "start_time_ms": min(times),
        "end_time_ms": max(times),
        "vehicle_count": len(times),
        "mean_headway": float(np.mean(headways)) if len(headways) else "",
    }


def infer_cycles(burst_rows: list[dict[str, Any]], cfg: InferenceConfig) -> list[dict[str, Any]]:
    if not burst_rows:
        return []
    df = pd.DataFrame(burst_rows)
    cycle_rows: list[dict[str, Any]] = []
    for key, group in df.sort_values("start_time_ms").groupby(["city", "record_name", "stopline_id", "movement"], sort=True):
        starts_s = group["start_time_ms"].astype(float).to_numpy() / 1000.0
        intervals = np.diff(starts_s)
        valid = intervals[(intervals >= cfg.min_cycle_s) & (intervals <= cfg.max_cycle_s)]
        cycle_s = float(np.median(valid)) if len(valid) else ""
        confidence = "high" if len(valid) >= 3 else "medium" if len(valid) >= 1 else "low"
        cycle_rows.append(
            {
                "city": key[0],
                "record_name": key[1],
                "stopline_id": key[2],
                "movement": key[3],
                "estimated_cycle_s": cycle_s,
                "autocorr_peak_s": cycle_s,
                "num_bursts": int(len(group)),
                "confidence": confidence,
            }
        )
    return cycle_rows


def infer_signal_windows(
    burst_rows: list[dict[str, Any]],
    cycle_rows: list[dict[str, Any]],
    cfg: InferenceConfig,
) -> list[dict[str, Any]]:
    if not burst_rows:
        return []
    cycles = {
        (row["city"], row["record_name"], row["stopline_id"], row["movement"]): row
        for row in cycle_rows
    }
    df = pd.DataFrame(burst_rows)
    window_rows: list[dict[str, Any]] = []
    for key, group in df.sort_values("start_time_ms").groupby(["city", "record_name", "stopline_id", "movement"], sort=True):
        rows = list(group.to_dict("records"))
        cycle = cycles.get(key, {})
        cycle_s = cycle.get("estimated_cycle_s", "")
        for idx, row in enumerate(rows):
            green_start = float(row["start_time_ms"])
            green_end = float(row["end_time_ms"])
            if idx + 1 < len(rows):
                red_end = float(rows[idx + 1]["start_time_ms"])
            elif cycle_s != "":
                red_end = green_start + float(cycle_s) * 1000.0
            else:
                red_end = ""
            red_start = ""
            window_rows.append(
                {
                    "city": key[0],
                    "record_name": key[1],
                    "stopline_id": key[2],
                    "movement": key[3],
                    "green_start_ms": green_start,
                    "green_end_ms": green_end,
                    "green_duration_ms": max(0.0, green_end - green_start),
                    "red_start_ms": red_start,
                    "red_end_ms": red_end,
                    "red_duration_ms": "",
                    "cycle_s": cycle_s,
                    "confidence": cycle.get("confidence", "low"),
                }
            )
    return window_rows


CROSSING_EVENT_FIELDS = [
    "city",
    "record_name",
    "track_id",
    "vehicle_global_id",
    "stopline_id",
    "osm_way_id",
    "crossing_frame_before",
    "crossing_timestamp_ms",
    "crossing_x",
    "crossing_y",
    "speed",
    "heading_before",
    "heading_final",
    "delta_heading",
    "movement",
]

MOVEMENT_BURST_FIELDS = [
    "city",
    "record_name",
    "stopline_id",
    "movement",
    "burst_id",
    "start_time_ms",
    "end_time_ms",
    "vehicle_count",
    "mean_headway",
]

CYCLE_INFERENCE_FIELDS = [
    "city",
    "record_name",
    "stopline_id",
    "movement",
    "estimated_cycle_s",
    "autocorr_peak_s",
    "num_bursts",
    "confidence",
]

MOVEMENT_SIGNAL_WINDOW_FIELDS = [
    "city",
    "record_name",
    "stopline_id",
    "movement",
    "green_start_ms",
    "green_end_ms",
    "green_duration_ms",
    "red_start_ms",
    "red_end_ms",
    "red_duration_ms",
    "cycle_s",
    "confidence",
]
