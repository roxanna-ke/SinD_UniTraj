from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from metadrive.scenario import ScenarioDescription as SD
from metadrive.type import MetaDriveType
from scenarionet.converter.utils import write_to_directory_single_worker


TRACK_LENGTH = 81
DEFAULT_VEHICLE_HEIGHT = 1.5
DEFAULT_PEDESTRIAN_LENGTH = 0.5
DEFAULT_PEDESTRIAN_WIDTH = 0.5
DEFAULT_PEDESTRIAN_HEIGHT = 1.7
VEHICLE_CLASSES = {"car", "truck", "bus"}
CYCLIST_CLASSES = {"bicycle", "motorcycle", "tricycle"}
PEDESTRIAN_CLASSES = {"pedestrian"}
TRAFFIC_LIGHT_STATE_MAP = {
    0: MetaDriveType.LANE_STATE_UNKNOWN,
    1: MetaDriveType.LANE_STATE_STOP,
    2: MetaDriveType.LANE_STATE_CAUTION,
    3: MetaDriveType.LANE_STATE_GO,
}
WAY_TYPE_TO_SCENARIONET = {
    "curbstone": "ROAD_EDGE_BOUNDARY",
    "line_thin": "ROAD_LINE_SOLID_SINGLE_WHITE",
    "stop_line": "ROAD_LINE_SOLID_SINGLE_WHITE",
    "virtual": "ROAD_LINE_BROKEN_SINGLE_WHITE",
}


@dataclass(frozen=True)
class SindWindow:
    city: str
    record_name: str
    focal_track_id: str
    start_index: int
    timestamps_ms: tuple[float, ...]
    track_rows: pd.DataFrame
    pedestrian_rows: pd.DataFrame
    map_features: dict[str, dict[str, Any]]
    dynamic_map_states: dict[str, dict[str, Any]]

    @property
    def scenario_id(self) -> str:
        return f"{self.city.lower()}_{self.record_name}_{self.focal_track_id}_{self.start_index:05d}"


@dataclass(frozen=True)
class SindRecord:
    city: str
    record_name: str
    vehicle_tracks: pd.DataFrame
    vehicle_meta: pd.DataFrame
    pedestrian_tracks: pd.DataFrame
    pedestrian_meta: pd.DataFrame
    traffic_light: pd.DataFrame
    map_features: dict[str, dict[str, Any]]
    lane_centers: dict[str, np.ndarray]
    recording_meta: pd.Series


def _normalize_track_id(value: object) -> str:
    return str(value)


def _infer_object_type(agent_class: str) -> str:
    normalized = str(agent_class).strip().lower()
    if normalized in VEHICLE_CLASSES:
        return MetaDriveType.VEHICLE
    if normalized in CYCLIST_CLASSES:
        return MetaDriveType.CYCLIST
    if normalized in PEDESTRIAN_CLASSES:
        return MetaDriveType.PEDESTRIAN
    return MetaDriveType.OTHER


def _project_lon_lat_to_xy(lon: float, lat: float) -> np.ndarray:
    meters_per_degree = 111_320.0
    return np.array([lon * meters_per_degree, lat * meters_per_degree, 0.0], dtype=np.float32)


def _polyline_from_way(node_lookup: dict[str, np.ndarray], way_elem: ET.Element) -> np.ndarray:
    points = []
    for node_ref in way_elem.findall("nd"):
        node_id = node_ref.attrib["ref"]
        if node_id in node_lookup:
            points.append(node_lookup[node_id])
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _average_centerline(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    num_points = min(len(left), len(right))
    if num_points == 0:
        return np.zeros((0, 3), dtype=np.float32)
    left_trim = left[:num_points]
    right_trim = right[:num_points]
    return (left_trim + right_trim) / 2.0


def _polygon_from_boundaries(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    num_points = min(len(left), len(right))
    if num_points == 0:
        return np.zeros((0, 3), dtype=np.float32)
    left_trim = left[:num_points]
    right_trim = right[:num_points]
    return np.concatenate([left_trim, right_trim[::-1]], axis=0)


def _parse_lanelet_map(city_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray]]:
    osm_path = city_dir / "map_relink_law_save.osm"
    root = ET.parse(osm_path).getroot()

    node_lookup = {
        node.attrib["id"]: _project_lon_lat_to_xy(float(node.attrib["lon"]), float(node.attrib["lat"]))
        for node in root.findall("node")
    }

    way_lookup: dict[str, ET.Element] = {way.attrib["id"]: way for way in root.findall("way")}
    way_polylines = {way_id: _polyline_from_way(node_lookup, way) for way_id, way in way_lookup.items()}

    map_features: dict[str, dict[str, Any]] = {}
    lane_centers: dict[str, np.ndarray] = {}
    stop_lines: dict[str, np.ndarray] = {}

    for way_id, way_elem in way_lookup.items():
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way_elem.findall("tag")}
        way_type = tags.get("type")
        polyline = way_polylines[way_id]
        if len(polyline) == 0:
            continue
        if way_type == "traffic_light":
            stop_lines[way_id] = polyline
            continue
        mapped_type = WAY_TYPE_TO_SCENARIONET.get(way_type)
        if mapped_type:
            feature_name = f"boundary_{way_id}"
            map_features[feature_name] = {"type": mapped_type, "polyline": polyline}

    for relation in root.findall("relation"):
        members = {member.attrib["role"]: member.attrib["ref"] for member in relation.findall("member")}
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in relation.findall("tag")}
        if tags.get("type") != "lanelet":
            continue
        left_id = members.get("left")
        right_id = members.get("right")
        left = way_polylines.get(left_id) if left_id is not None else None
        right = way_polylines.get(right_id) if right_id is not None else None
        if left is None or right is None or len(left) == 0 or len(right) == 0:
            continue
        feature_name = tags.get("name", relation.attrib["id"])
        if tags.get("subtype") == "crosswalk":
            map_features[f"crosswalk_{feature_name}"] = {
                "type": MetaDriveType.CROSSWALK,
                "polygon": _polygon_from_boundaries(left, right),
            }
            continue
        centerline = _average_centerline(left, right)
        lane_centers[feature_name] = centerline
        map_features[f"lane_{feature_name}"] = {
            "type": MetaDriveType.LANE_SURFACE_STREET,
            "polyline": centerline,
            "polygon": _polygon_from_boundaries(left, right),
            "entry_lanes": [],
            "exit_lanes": [],
        }

    if lane_centers:
        lane_items = list(lane_centers.items())
        for index, (lane_name, _) in enumerate(lane_items):
            lane_feature = map_features[f"lane_{lane_name}"]
            if index > 0:
                lane_feature["entry_lanes"] = [f"lane_{lane_items[index - 1][0]}"]
            if index + 1 < len(lane_items):
                lane_feature["exit_lanes"] = [f"lane_{lane_items[index + 1][0]}"]

    return map_features, lane_centers, stop_lines


def _load_record(sind_record_dir: Path, city: str) -> SindRecord:
    vehicle_tracks = pd.read_csv(sind_record_dir / "Veh_smoothed_tracks.csv")
    vehicle_meta = pd.read_csv(sind_record_dir / "Veh_tracks_meta.csv")
    pedestrian_tracks = pd.read_csv(sind_record_dir / "Ped_smoothed_tracks.csv")
    pedestrian_meta = pd.read_csv(sind_record_dir / "Ped_tracks_meta.csv")
    traffic_light = pd.read_csv(sind_record_dir / f"TrafficLight_{sind_record_dir.name}.csv")
    recording_meta = pd.read_csv(sind_record_dir / "recording_metas.csv").iloc[0]
    map_features, lane_centers, _stop_lines = _parse_lanelet_map(sind_record_dir.parent)
    return SindRecord(
        city=city,
        record_name=sind_record_dir.name,
        vehicle_tracks=vehicle_tracks,
        vehicle_meta=vehicle_meta,
        pedestrian_tracks=pedestrian_tracks,
        pedestrian_meta=pedestrian_meta,
        traffic_light=traffic_light,
        map_features=map_features,
        lane_centers=lane_centers,
        recording_meta=recording_meta,
    )


def _traffic_lights_for_window(record: SindRecord, timestamps_ms: np.ndarray) -> dict[str, dict[str, Any]]:
    if not record.lane_centers:
        return {}
    lane_names = list(record.lane_centers.keys())
    lane_cycle = lane_names[:8] if len(lane_names) >= 8 else [lane_names[i % len(lane_names)] for i in range(8)]
    signal_columns = [col for col in record.traffic_light.columns if col.startswith("Traffic light ")]
    traffic_df = record.traffic_light.sort_values("timestamp(ms)")
    signal_states: dict[str, dict[str, Any]] = {}
    for idx, col in enumerate(signal_columns):
        lane_name = lane_cycle[idx]
        lane_center = record.lane_centers[lane_name]
        stop_point = lane_center[len(lane_center) // 2] if len(lane_center) else np.zeros(3, dtype=np.float32)
        object_states = []
        raw_times = traffic_df["timestamp(ms)"].to_numpy(dtype=float)
        raw_values = traffic_df[col].to_numpy(dtype=int)
        for ts in timestamps_ms:
            state_idx = np.searchsorted(raw_times, ts, side="right") - 1
            if state_idx < 0:
                raw_state = 0
            else:
                raw_state = int(raw_values[state_idx])
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


def _eligible_windows(record: SindRecord, max_scenarios: int | None, stride: int) -> list[SindWindow]:
    vehicle_meta = record.vehicle_meta.copy()
    vehicle_meta["trackId"] = vehicle_meta["trackId"].map(_normalize_track_id)

    pedestrian_tracks = record.pedestrian_tracks.copy()
    pedestrian_tracks["track_id"] = pedestrian_tracks["track_id"].map(_normalize_track_id)

    windows: list[SindWindow] = []
    for _, meta_row in vehicle_meta.iterrows():
        track_class = str(meta_row["class"]).strip().lower()
        if track_class not in VEHICLE_CLASSES:
            continue

        track_id = _normalize_track_id(meta_row["trackId"])
        track_rows = record.vehicle_tracks[record.vehicle_tracks["track_id"].map(_normalize_track_id) == track_id].copy()
        track_rows = track_rows.sort_values("frame_id")
        if len(track_rows) < TRACK_LENGTH:
            continue

        frame_ids = track_rows["frame_id"].to_numpy(dtype=int)
        timestamps_ms = track_rows["timestamp_ms"].to_numpy(dtype=float)
        focal_meta = track_rows.set_index("frame_id")
        for start_pos in range(0, len(track_rows) - TRACK_LENGTH + 1, stride):
            end_pos = start_pos + TRACK_LENGTH
            expected = np.arange(frame_ids[start_pos], frame_ids[start_pos] + TRACK_LENGTH)
            window_frames = frame_ids[start_pos:end_pos]
            if not np.array_equal(window_frames, expected):
                continue
            current_frame = int(window_frames[20])
            if current_frame not in focal_meta.index:
                continue
            start_frame = int(window_frames[0])
            end_frame = int(window_frames[-1])
            timestamps_window = np.asarray(timestamps_ms[start_pos:end_pos], dtype=float)
            ped_rows = pedestrian_tracks[pedestrian_tracks["frame_id"].between(start_frame, end_frame)].copy()
            windows.append(
                SindWindow(
                    city=record.city,
                    record_name=record.record_name,
                    focal_track_id=track_id,
                    start_index=start_frame,
                    timestamps_ms=tuple(float(x) for x in timestamps_window),
                    track_rows=record.vehicle_tracks[
                        record.vehicle_tracks["frame_id"].between(start_frame, end_frame)
                    ].copy(),
                    pedestrian_rows=ped_rows,
                    map_features=record.map_features,
                    dynamic_map_states=_traffic_lights_for_window(record, timestamps_window),
                )
            )
            if max_scenarios is not None and len(windows) >= max_scenarios:
                return windows
    return windows


def _build_track_state(track_df: pd.DataFrame, timestamps_ms: np.ndarray, object_type: str) -> dict[str, np.ndarray]:
    track_df = track_df.copy().sort_values("frame_id")
    time_index = {float(ts): idx for idx, ts in enumerate(timestamps_ms)}
    position = np.zeros((TRACK_LENGTH, 3), dtype=np.float32)
    heading = np.zeros((TRACK_LENGTH,), dtype=np.float32)
    velocity = np.zeros((TRACK_LENGTH, 2), dtype=np.float32)
    valid = np.zeros((TRACK_LENGTH,), dtype=np.float32)
    if object_type == MetaDriveType.PEDESTRIAN:
        length = np.full((TRACK_LENGTH, 1), DEFAULT_PEDESTRIAN_LENGTH, dtype=np.float32)
        width = np.full((TRACK_LENGTH, 1), DEFAULT_PEDESTRIAN_WIDTH, dtype=np.float32)
        height = np.full((TRACK_LENGTH, 1), DEFAULT_PEDESTRIAN_HEIGHT, dtype=np.float32)
    else:
        length = np.zeros((TRACK_LENGTH, 1), dtype=np.float32)
        width = np.zeros((TRACK_LENGTH, 1), dtype=np.float32)
        height = np.full((TRACK_LENGTH, 1), DEFAULT_VEHICLE_HEIGHT, dtype=np.float32)

    for _, row in track_df.iterrows():
        idx = time_index.get(float(row["timestamp_ms"]))
        if idx is None:
            idx = time_index.get(float(np.float32(row["timestamp_ms"])))
        if idx is None:
            continue
        position[idx] = np.array([row["x"], row["y"], 0.0], dtype=np.float32)
        if "yaw_rad" in row.index and not pd.isna(row["yaw_rad"]):
            heading[idx] = np.float32(row["yaw_rad"])
        else:
            vx = float(row.get("vx", 0.0))
            vy = float(row.get("vy", 0.0))
            heading[idx] = np.float32(math.atan2(vy, vx)) if abs(vx) + abs(vy) > 1e-4 else 0.0
        velocity[idx] = np.array([row.get("vx", 0.0), row.get("vy", 0.0)], dtype=np.float32)
        valid[idx] = 1.0
        if object_type != MetaDriveType.PEDESTRIAN:
            length[idx, 0] = np.float32(row["length"])
            width[idx, 0] = np.float32(row["width"])

    return {
        "position": position,
        "heading": heading,
        "velocity": velocity,
        "valid": valid,
        "length": length,
        "width": width,
        "height": height,
    }


def convert_sind_window(window: SindWindow, dataset_version: str, dataset_name: str = "sind") -> SD:
    scenario = SD()
    scenario[SD.ID] = window.scenario_id
    scenario[SD.VERSION] = f"{dataset_name}_{dataset_version}"
    scenario[SD.LENGTH] = TRACK_LENGTH

    timestamps_ms = np.asarray(window.timestamps_ms, dtype=np.float32)
    timestamps_s = (timestamps_ms - timestamps_ms[0]) / 1000.0

    combined_tracks = [window.track_rows.copy(), window.pedestrian_rows.copy()]
    track_groups = pd.concat(combined_tracks, ignore_index=True)
    track_groups["track_id"] = track_groups["track_id"].map(_normalize_track_id)

    tracks = {}
    ordered_track_ids: list[str] = []
    for track_id, track_df in track_groups.groupby("track_id", sort=True):
        ordered_track_ids.append(track_id)
        track_class = str(track_df.iloc[0]["agent_type"]).strip().lower()
        object_type = _infer_object_type(track_class)
        tracks[track_id] = {
            "type": object_type,
            "state": _build_track_state(track_df, timestamps_ms, object_type),
            "metadata": {
                "track_length": TRACK_LENGTH,
                "type": track_class,
                "object_id": track_id,
                "dataset": dataset_name,
            },
        }

    scenario[SD.TRACKS] = tracks
    scenario[SD.DYNAMIC_MAP_STATES] = window.dynamic_map_states
    scenario[SD.MAP_FEATURES] = window.map_features

    focal_track_index = ordered_track_ids.index(window.focal_track_id)
    scenario[SD.METADATA] = {
        SD.ID: scenario[SD.ID],
        "scenario_id": scenario[SD.ID],
        "dataset": dataset_name,
        SD.COORDINATE: MetaDriveType.COORDINATE_METADRIVE,
        SD.TIMESTEP: timestamps_s.astype(np.float32),
        "sample_rate": float(np.median(np.diff(timestamps_s))) if len(timestamps_s) > 1 else 0.1,
        SD.METADRIVE_PROCESSED: False,
        SD.SDC_ID: window.focal_track_id,
        "source_file": window.record_name,
        "track_length": TRACK_LENGTH,
        "current_time_index": 20,
        "sdc_track_index": focal_track_index,
        "tracks_to_predict": {
            window.focal_track_id: {
                "track_index": focal_track_index,
                "track_id": window.focal_track_id,
                "difficulty": 0,
                "object_type": tracks[window.focal_track_id]["type"],
            }
        },
    }
    return scenario


def _write_split(windows: list[SindWindow], output_dir: Path, dataset_name: str, dataset_version: str) -> None:
    write_to_directory_single_worker(
        convert_func=convert_sind_window,
        scenarios=windows,
        output_path=str(output_dir),
        dataset_version=dataset_version,
        dataset_name=dataset_name,
        overwrite=True,
        worker_index=0,
        report_memory_freq=None,
        preprocess=lambda scenarios, worker_index: scenarios,
    )


def convert_sind_record(
    sind_record_dir: Path,
    output_dir: Path,
    city: str,
    dataset_name: str = "sind",
    dataset_version: str = "v1",
    past_len: int = 21,
    future_len: int = 60,
    stride: int = 20,
    max_scenarios: int | None = None,
    train_ratio: float = 0.8,
) -> dict[str, Path]:
    if past_len + future_len != TRACK_LENGTH:
        raise ValueError(f"This converter expects {TRACK_LENGTH} total steps, got {past_len + future_len}")

    record = _load_record(Path(sind_record_dir), city=city)
    windows = _eligible_windows(record, max_scenarios=max_scenarios, stride=stride)
    if len(windows) < 2:
        raise ValueError("Not enough eligible SinD windows were generated")

    split_index = max(1, min(len(windows) - 1, int(len(windows) * train_ratio)))
    train_windows = windows[:split_index]
    val_windows = windows[split_index:]
    if not val_windows:
        val_windows = windows[-1:]
        train_windows = windows[:-1]

    train_dir = Path(output_dir) / "train" / dataset_name
    val_dir = Path(output_dir) / "val" / dataset_name
    _write_split(train_windows, train_dir, dataset_name=dataset_name, dataset_version=dataset_version)
    _write_split(val_windows, val_dir, dataset_name=dataset_name, dataset_version=dataset_version)
    return {"train": train_dir, "val": val_dir}


def convert_sind_dataset(
    sind_data_root: Path,
    output_dir: Path,
    dataset_name: str = "sind",
    dataset_version: str = "v1",
    cities: list[str] | None = None,
    max_records: int | None = None,
    stride: int = 20,
    max_scenarios_per_record: int | None = None,
    train_ratio: float = 0.8,
) -> dict[str, Path]:
    data_root = Path(sind_data_root)
    city_dirs = [data_root / city for city in cities] if cities else sorted([p for p in data_root.iterdir() if p.is_dir()])
    record_dirs = []
    for city_dir in city_dirs:
        record_dirs.extend(sorted([p for p in city_dir.iterdir() if p.is_dir()]))
    if len(record_dirs) < 2:
        raise ValueError("Need at least two SinD record directories for dataset conversion")

    records_with_windows: list[tuple[Path, list[SindWindow]]] = []
    for record_dir in record_dirs:
        required_files = [
            record_dir / "Veh_smoothed_tracks.csv",
            record_dir / "Veh_tracks_meta.csv",
            record_dir / "Ped_smoothed_tracks.csv",
            record_dir / "Ped_tracks_meta.csv",
            record_dir / f"TrafficLight_{record_dir.name}.csv",
            record_dir / "recording_metas.csv",
        ]
        if not all(path.exists() for path in required_files):
            continue
        record = _load_record(record_dir, city=record_dir.parent.name)
        windows = _eligible_windows(record, max_scenarios=max_scenarios_per_record, stride=stride)
        if windows:
            records_with_windows.append((record_dir, windows))
            if max_records is not None and len(records_with_windows) >= max_records:
                break
    if len(records_with_windows) < 2:
        raise ValueError("Need at least two non-empty SinD records after window generation")

    split_index = max(1, min(len(records_with_windows) - 1, int(len(records_with_windows) * train_ratio)))
    train_records = records_with_windows[:split_index]
    val_records = records_with_windows[split_index:]
    if not val_records:
        val_records = records_with_windows[-1:]
        train_records = records_with_windows[:-1]

    train_windows = [window for _, windows in train_records for window in windows]
    val_windows = [window for _, windows in val_records for window in windows]
    train_dir = Path(output_dir) / "train" / dataset_name
    val_dir = Path(output_dir) / "val" / dataset_name
    _write_split(train_windows, train_dir, dataset_name=dataset_name, dataset_version=dataset_version)
    _write_split(val_windows, val_dir, dataset_name=dataset_name, dataset_version=dataset_version)
    return {"train": train_dir, "val": val_dir}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SinD data into a ScenarioNet-style dataset")
    parser.add_argument("--sind-record-dir", type=Path)
    parser.add_argument("--sind-data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--city", type=str, default="Tianjin")
    parser.add_argument("--dataset-name", type=str, default="sind")
    parser.add_argument("--dataset-version", type=str, default="v1")
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--max-scenarios", type=int, default=None)
    parser.add_argument("--max-scenarios-per-record", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--cities", nargs="*", default=None)
    args = parser.parse_args()

    if args.sind_data_root is not None:
        convert_sind_dataset(
            sind_data_root=args.sind_data_root,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            cities=args.cities,
            max_records=args.max_records,
            stride=args.stride,
            max_scenarios_per_record=args.max_scenarios_per_record,
            train_ratio=args.train_ratio,
        )
    elif args.sind_record_dir is not None:
        convert_sind_record(
            sind_record_dir=args.sind_record_dir,
            output_dir=args.output_dir,
            city=args.city,
            dataset_name=args.dataset_name,
            dataset_version=args.dataset_version,
            stride=args.stride,
            max_scenarios=args.max_scenarios,
            train_ratio=args.train_ratio,
        )
    else:
        raise ValueError("Provide either --sind-record-dir or --sind-data-root")


if __name__ == "__main__":
    main()
