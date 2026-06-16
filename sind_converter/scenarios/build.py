from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType

from sind_converter.lights.standardize import dynamic_map_states_for_window


DEFAULT_VEHICLE_HEIGHT = 1.5
DEFAULT_PEDESTRIAN_LENGTH = 0.5
DEFAULT_PEDESTRIAN_WIDTH = 0.5
DEFAULT_PEDESTRIAN_HEIGHT = 1.7
VEHICLE_CLASSES = {"car", "truck", "bus"}
CYCLIST_CLASSES = {"bicycle", "motorcycle", "tricycle"}
PEDESTRIAN_CLASSES = {"pedestrian"}


@dataclass(frozen=True)
class ScenarioWindow:
    city: str
    record_name: str
    start_frame: int
    timestamps_ms: tuple[float, ...]
    vehicle_rows: pd.DataFrame
    pedestrian_rows: pd.DataFrame
    target_ids: tuple[str, ...]
    map_features: dict[str, dict[str, Any]]
    lane_centers: dict[str, np.ndarray]
    traffic_light: pd.DataFrame | None
    past_len: int
    future_len: int

    @property
    def scenario_id(self) -> str:
        return f"sind_{self.city.lower()}_{self.record_name}_{self.start_frame:06d}"

    @property
    def total_length(self) -> int:
        return self.past_len + self.future_len


def infer_object_type(agent_class: str) -> str:
    normalized = str(agent_class).strip().lower()
    if normalized in VEHICLE_CLASSES:
        return MetaDriveType.VEHICLE
    if normalized in CYCLIST_CLASSES:
        return MetaDriveType.CYCLIST
    if normalized in PEDESTRIAN_CLASSES:
        return MetaDriveType.PEDESTRIAN
    return MetaDriveType.OTHER


def _valid_full_window(track_df: pd.DataFrame, frames: np.ndarray) -> bool:
    frame_set = set(track_df["frame_id"].to_numpy(dtype=int))
    return all(int(frame) in frame_set for frame in frames)


def generate_windows(
    city: str,
    record_name: str,
    vehicle_tracks: pd.DataFrame,
    pedestrian_tracks: pd.DataFrame,
    map_features: dict[str, dict[str, Any]],
    lane_centers: dict[str, np.ndarray],
    traffic_light: pd.DataFrame | None,
    past_len: int,
    future_len: int,
    stride: int,
    max_scenarios: int | None = None,
) -> list[ScenarioWindow]:
    total_length = past_len + future_len
    sorted_frames = np.array(sorted(vehicle_tracks["frame_id"].dropna().astype(int).unique()))
    if len(sorted_frames) < total_length:
        return []

    windows: list[ScenarioWindow] = []
    vehicle_tracks = vehicle_tracks.copy()
    pedestrian_tracks = pedestrian_tracks.copy()
    vehicle_tracks["track_id"] = vehicle_tracks["track_id"].map(str)
    pedestrian_tracks["track_id"] = pedestrian_tracks["track_id"].map(str)
    vehicle_groups = {track_id: group.sort_values("frame_id") for track_id, group in vehicle_tracks.groupby("track_id", sort=True)}

    for start_pos in range(0, len(sorted_frames) - total_length + 1, stride):
        frames = sorted_frames[start_pos : start_pos + total_length]
        if not np.array_equal(frames, np.arange(frames[0], frames[0] + total_length)):
            continue
        start_frame = int(frames[0])
        end_frame = int(frames[-1])
        center_frame = int(frames[past_len - 1])
        targets = []
        for track_id, track_df in vehicle_groups.items():
            agent_class = str(track_df.iloc[0].get("agent_type", "")).strip().lower()
            if agent_class not in VEHICLE_CLASSES:
                continue
            if center_frame not in set(track_df["frame_id"].to_numpy(dtype=int)):
                continue
            if _valid_full_window(track_df, frames):
                targets.append(track_id)
        if not targets:
            continue

        timestamps_ms = (
            vehicle_tracks[vehicle_tracks["frame_id"].isin(frames)]
            .groupby("frame_id", sort=True)["timestamp_ms"]
            .first()
            .reindex(frames)
            .to_numpy(dtype=float)
        )
        vehicle_rows = vehicle_tracks[vehicle_tracks["frame_id"].between(start_frame, end_frame)].copy()
        pedestrian_rows = pedestrian_tracks[pedestrian_tracks["frame_id"].between(start_frame, end_frame)].copy()
        windows.append(
            ScenarioWindow(
                city=city,
                record_name=record_name,
                start_frame=start_frame,
                timestamps_ms=tuple(float(x) for x in timestamps_ms),
                vehicle_rows=vehicle_rows,
                pedestrian_rows=pedestrian_rows,
                target_ids=tuple(sorted(targets, key=str)),
                map_features=map_features,
                lane_centers=lane_centers,
                traffic_light=traffic_light,
                past_len=past_len,
                future_len=future_len,
            )
        )
        if max_scenarios is not None and len(windows) >= max_scenarios:
            return windows
    return windows


def build_track_state(track_df: pd.DataFrame, timestamps_ms: np.ndarray, object_type: str, total_length: int) -> dict[str, np.ndarray]:
    track_df = track_df.copy().sort_values("frame_id")
    time_index = {float(ts): idx for idx, ts in enumerate(timestamps_ms)}
    position = np.zeros((total_length, 3), dtype=np.float32)
    heading = np.zeros((total_length,), dtype=np.float32)
    velocity = np.zeros((total_length, 2), dtype=np.float32)
    valid = np.zeros((total_length,), dtype=np.float32)
    if object_type == MetaDriveType.PEDESTRIAN:
        length = np.full((total_length, 1), DEFAULT_PEDESTRIAN_LENGTH, dtype=np.float32)
        width = np.full((total_length, 1), DEFAULT_PEDESTRIAN_WIDTH, dtype=np.float32)
        height = np.full((total_length, 1), DEFAULT_PEDESTRIAN_HEIGHT, dtype=np.float32)
    else:
        length = np.zeros((total_length, 1), dtype=np.float32)
        width = np.zeros((total_length, 1), dtype=np.float32)
        height = np.full((total_length, 1), DEFAULT_VEHICLE_HEIGHT, dtype=np.float32)

    for _, row in track_df.iterrows():
        idx = time_index.get(float(row["timestamp_ms"]))
        if idx is None:
            continue
        vx = float(row.get("vx", 0.0))
        vy = float(row.get("vy", 0.0))
        position[idx] = np.array([row["x"], row["y"], 0.0], dtype=np.float32)
        if "yaw_rad" in row.index and not pd.isna(row["yaw_rad"]):
            heading[idx] = np.float32(row["yaw_rad"])
        elif "heading_rad" in row.index and not pd.isna(row["heading_rad"]):
            heading[idx] = np.float32(row["heading_rad"])
        else:
            heading[idx] = np.float32(math.atan2(vy, vx)) if abs(vx) + abs(vy) > 1e-4 else 0.0
        velocity[idx] = np.array([vx, vy], dtype=np.float32)
        valid[idx] = 1.0
        if object_type != MetaDriveType.PEDESTRIAN:
            length[idx, 0] = np.float32(row.get("length", 0.0))
            width[idx, 0] = np.float32(row.get("width", 0.0))

    return {"position": position, "length": length, "width": width, "height": height, "heading": heading, "velocity": velocity, "valid": valid}


def convert_window_to_scenario(window: ScenarioWindow, dataset_version: str, dataset_name: str = "sind") -> SD:
    scenario = SD()
    scenario[SD.ID] = window.scenario_id
    scenario[SD.VERSION] = f"{dataset_name}_{dataset_version}"
    scenario[SD.LENGTH] = window.total_length

    timestamps_ms = np.asarray(window.timestamps_ms, dtype=np.float32)
    timestamps_s = (timestamps_ms - timestamps_ms[0]) / 1000.0
    combined_tracks = [window.vehicle_rows.copy(), window.pedestrian_rows.copy()]
    track_groups = pd.concat(combined_tracks, ignore_index=True)
    track_groups["track_id"] = track_groups["track_id"].map(str)

    tracks = {}
    ordered_track_ids: list[str] = []
    for track_id, track_df in track_groups.groupby("track_id", sort=True):
        ordered_track_ids.append(track_id)
        track_class = str(track_df.iloc[0].get("agent_type", "unknown")).strip().lower()
        object_type = infer_object_type(track_class)
        tracks[track_id] = {
            "type": object_type,
            "state": build_track_state(track_df, timestamps_ms, object_type, window.total_length),
            "metadata": {"track_length": window.total_length, "type": track_class, "object_id": track_id, "dataset": dataset_name},
        }

    sdc_id = window.target_ids[0]
    scenario[SD.TRACKS] = tracks
    scenario[SD.DYNAMIC_MAP_STATES] = dynamic_map_states_for_window(window.traffic_light, window.lane_centers, timestamps_ms)
    scenario[SD.MAP_FEATURES] = window.map_features
    tracks_to_predict = {}
    for target_id in window.target_ids:
        if target_id not in tracks:
            continue
        tracks_to_predict[target_id] = {
            "track_index": ordered_track_ids.index(target_id),
            "track_id": target_id,
            "difficulty": 0,
            "object_type": tracks[target_id]["type"],
        }

    scenario[SD.METADATA] = {
        SD.ID: scenario[SD.ID],
        "scenario_id": scenario[SD.ID],
        "dataset": dataset_name,
        SD.COORDINATE: MetaDriveType.COORDINATE_METADRIVE,
        SD.TIMESTEP: timestamps_s.astype(np.float32),
        "sample_rate": float(np.median(np.diff(timestamps_s))) if len(timestamps_s) > 1 else 0.1,
        SD.METADRIVE_PROCESSED: False,
        SD.SDC_ID: sdc_id,
        "source_file": window.record_name,
        "record_name": window.record_name,
        "city": window.city,
        "track_length": window.total_length,
        "current_time_index": window.past_len - 1,
        "sdc_track_index": ordered_track_ids.index(sdc_id),
        "tracks_to_predict": tracks_to_predict,
        "split_candidates": {"record": window.record_name, "city": window.city},
    }
    return scenario
