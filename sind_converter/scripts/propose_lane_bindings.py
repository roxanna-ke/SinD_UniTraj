from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIONET_ROOT = PROJECT_ROOT / "scenarionet"
for path in (PROJECT_ROOT, SCENARIONET_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sind_converter.data.discovery import resolve_map_path
from sind_converter.lights.stopline_extraction import StopLine, extract_stop_lines
from sind_converter.maps.osm import parse_osm_map


DIRECT_APPROACH_PATTERNS = (
    re.compile(r"^(?P<arm>[WNSE])_en_(?P<lane>\d+)$", flags=re.IGNORECASE),
    re.compile(r"^(?P<arm>R\d+)In(?P<lane>\d+)$", flags=re.IGNORECASE),
)
CONNECTOR_PATTERNS = (
    re.compile(r"^(?P<src>[WNSE])_EX_\d+_to_(?P<dst>[WNSE])_EN_\d+$", flags=re.IGNORECASE),
    re.compile(r"^(?P<src>R\d+)In\d+_to_(?P<dst>R\d+)Out\d+$", flags=re.IGNORECASE),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate lane_bindings JSON files from SinD maps.")
    parser.add_argument("--data-root", type=Path, default=Path("SinD/Dataset"))
    parser.add_argument("--map-fallback-root", type=Path, default=Path("SinD/Data"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sind_converter/lights/config/lane_bindings"),
        help="Directory to write per-city JSON files",
    )
    parser.add_argument("--cities", nargs="*", default=["Xi_an", "Changchun", "Chongqing", "Tianjin"])
    parser.add_argument("--max-stopline-distance-m", type=float, default=18.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for city in args.cities:
        map_path = resolve_map_path(city, args.data_root, args.map_fallback_root)
        map_features, lane_centers = parse_osm_map(map_path)
        stop_lines = extract_stop_lines(map_path, city)
        payload = build_candidate_lane_bindings(city, map_features, lane_centers, stop_lines, args.max_stopline_distance_m)
        output_path = args.output_dir / f"{city}.json"
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        print(f"[ok] wrote {output_path} with {len(payload['lane_bindings'])} bindings")


def build_candidate_lane_bindings(
    city: str,
    map_features: dict[str, dict],
    lane_centers: dict[str, np.ndarray],
    stop_lines: list[StopLine],
    max_stopline_distance_m: float,
) -> dict:
    direct_approach_lanes = []
    connector_lanes = []
    for lane_name, centerline in lane_centers.items():
        if lane_name.endswith(":gap"):
            continue
        lane_feature_id = f"lane_{lane_name}"
        if lane_feature_id not in map_features:
            continue
        direct = _parse_direct_approach(lane_name)
        if direct is not None:
            direct_approach_lanes.append((lane_name, lane_feature_id, direct, centerline))
            continue
        connector = _parse_connector(lane_name)
        if connector is not None:
            connector_lanes.append((lane_name, lane_feature_id, connector, centerline))

    stopline_to_arm = _map_stoplines_to_arms(stop_lines, direct_approach_lanes, max_stopline_distance_m)
    connector_source_arms = sorted({source_arm for _, _, (source_arm, _), _ in connector_lanes})
    if connector_source_arms and set(connector_source_arms).issubset({"N", "E", "S", "W"}):
        connector_stopline_to_arm = _map_cardinal_stoplines(stop_lines)
    else:
        connector_stopline_to_arm = stopline_to_arm
    arm_cycle = _arm_cycle(stop_lines, connector_stopline_to_arm)

    bindings_by_key: dict[tuple[str, str], dict] = {}
    for lane_name, lane_feature_id, connector, centerline in connector_lanes:
        source_arm, dest_arm = connector
        stopline = connector_stopline_to_arm.get(source_arm)
        if stopline is None:
            continue
        movement = _classify_movement(source_arm, dest_arm, arm_cycle)
        if movement is None:
            continue
        stop_point = _nearest_point_to_stopline(centerline, stopline)
        key = (stopline.stopline_id, movement)
        row = bindings_by_key.setdefault(
            key,
            {
                "stopline_id": stopline.stopline_id,
                "movement": movement,
                "lane_ids": [],
                "stop_point": [round(float(stop_point[0]), 3), round(float(stop_point[1]), 3), round(float(stop_point[2]), 3)],
                "confidence": "medium",
                "source": "candidate_from_map_geometry",
                "notes": f"source_arm={source_arm}",
            },
        )
        row["lane_ids"].append(lane_feature_id)

    lane_bindings = []
    for (_, _), row in sorted(bindings_by_key.items()):
        row["lane_ids"] = sorted(set(row["lane_ids"]))
        if row["lane_ids"]:
            lane_bindings.append(row)
    return {
        "city": city,
        "lane_bindings": lane_bindings,
    }


def _parse_direct_approach(lane_name: str) -> str | None:
    for pattern in DIRECT_APPROACH_PATTERNS:
        match = pattern.match(lane_name)
        if match:
            return str(match.group("arm")).upper()
    return None


def _parse_connector(lane_name: str) -> tuple[str, str] | None:
    for pattern in CONNECTOR_PATTERNS:
        match = pattern.match(lane_name)
        if match:
            return str(match.group("src")).upper(), str(match.group("dst")).upper()
    return None


def _map_stoplines_to_arms(
    stop_lines: list[StopLine],
    direct_approach_lanes: list[tuple[str, str, str, np.ndarray]],
    max_stopline_distance_m: float,
) -> dict[str, StopLine]:
    known_arms = sorted({arm for _, _, arm, _ in direct_approach_lanes})
    if known_arms and set(known_arms).issubset({"N", "E", "S", "W"}):
        return _map_cardinal_stoplines(stop_lines)

    arm_votes: dict[str, list[tuple[float, StopLine]]] = defaultdict(list)
    for _, _, arm, centerline in direct_approach_lanes:
        point = _approach_endpoint_near_stopline(centerline, stop_lines)
        distances = [(_point_to_stopline_distance(point[:2], stopline), stopline) for stopline in stop_lines]
        distance, stopline = min(distances, key=lambda item: item[0])
        if distance <= max_stopline_distance_m:
            arm_votes[arm].append((distance, stopline))

    mapping: dict[str, StopLine] = {}
    for arm, votes in arm_votes.items():
        counts = Counter(stopline.stopline_id for _, stopline in votes)
        stopline_id = counts.most_common(1)[0][0]
        winner = min((item for item in votes if item[1].stopline_id == stopline_id), key=lambda item: item[0])[1]
        mapping[arm] = winner
    return mapping


def _map_cardinal_stoplines(stop_lines: list[StopLine]) -> dict[str, StopLine]:
    return {
        "N": max(stop_lines, key=lambda line: line.mid_y),
        "S": min(stop_lines, key=lambda line: line.mid_y),
        "E": max(stop_lines, key=lambda line: line.mid_x),
        "W": min(stop_lines, key=lambda line: line.mid_x),
    }


def _arm_cycle(stop_lines: list[StopLine], stopline_to_arm: dict[str, StopLine]) -> list[str]:
    intersection_center = np.mean(np.array([[line.mid_x, line.mid_y] for line in stop_lines], dtype=np.float64), axis=0)
    arm_angles = []
    for arm, stopline in stopline_to_arm.items():
        angle = math.atan2(stopline.mid_y - intersection_center[1], stopline.mid_x - intersection_center[0])
        arm_angles.append((angle, arm))
    arm_angles.sort()
    return [arm for _, arm in arm_angles]


def _classify_movement(source_arm: str, dest_arm: str, arm_cycle: list[str]) -> str | None:
    if source_arm not in arm_cycle or dest_arm not in arm_cycle or len(arm_cycle) != 4:
        return None
    src_idx = arm_cycle.index(source_arm)
    dst_idx = arm_cycle.index(dest_arm)
    diff = (dst_idx - src_idx) % 4
    if diff == 2:
        return "straight"
    if diff == 1:
        return "right"
    if diff == 3:
        return "left"
    return None


def _approach_endpoint_near_stopline(centerline: np.ndarray, stop_lines: list[StopLine]) -> np.ndarray:
    pts = np.asarray(centerline, dtype=np.float64)
    if len(pts) == 0:
        return np.zeros(3, dtype=np.float64)
    endpoints = [pts[0], pts[-1]]
    return min(
        endpoints,
        key=lambda point: min(_point_to_stopline_distance(point[:2], stopline) for stopline in stop_lines),
    )


def _nearest_point_to_stopline(centerline: np.ndarray, stopline: StopLine) -> np.ndarray:
    pts = np.asarray(centerline, dtype=np.float64)
    if len(pts) == 0:
        return np.array([stopline.mid_x, stopline.mid_y, 0.0], dtype=np.float64)
    distances = np.linalg.norm(pts[:, :2] - stopline.midpoint[None, :], axis=1)
    return pts[int(np.argmin(distances))]


def _point_to_stopline_distance(point_xy: np.ndarray, stopline: StopLine) -> float:
    seg = stopline.p2 - stopline.p1
    seg_norm = float(np.dot(seg, seg))
    if seg_norm <= 0.0:
        return float(np.linalg.norm(point_xy - stopline.p1))
    t = float(np.clip(np.dot(point_xy - stopline.p1, seg) / seg_norm, 0.0, 1.0))
    projection = stopline.p1 + t * seg
    return float(np.linalg.norm(point_xy - projection))


if __name__ == "__main__":
    main()
