from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


WAY_TYPE_TO_SCENARIONET = {
    "curbstone": "BOUNDARY_LINE",
    "line_thin": "ROAD_LINE_SOLID_SINGLE_WHITE",
    "line_thick": "ROAD_LINE_SOLID_SINGLE_WHITE",
    "stop_line": "ROAD_LINE_SOLID_SINGLE_WHITE",
    "virtual": "ROAD_LINE_BROKEN_SINGLE_WHITE",
}


def project_lon_lat_to_xy(lon: float, lat: float) -> np.ndarray:
    meters_per_degree = 111_320.0
    return np.array([lon * meters_per_degree, lat * meters_per_degree, 0.0], dtype=np.float32)


def audit_osm_tags(map_paths: list[Path], output_dir: Path) -> Path:
    inventory: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for map_path in map_paths:
        city = map_path.parent.name
        root = ET.parse(map_path).getroot()
        for elem_name in ["node", "way", "relation"]:
            for elem in root.findall(elem_name):
                for tag in elem.findall("tag"):
                    inventory[(elem_name, tag.attrib.get("k", ""), tag.attrib.get("v", ""))].add(city)

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "osm_stable_mapping_table.csv"
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["element", "tag_key", "tag_value", "cities", "scenarionet_feature_type", "confidence", "v1_strategy", "degrade_strategy"],
        )
        writer.writeheader()
        for (element, key, value), cities in sorted(inventory.items()):
            feature_type = ""
            strategy = "skip"
            confidence = "low"
            if element == "relation" and key == "type" and value == "lanelet":
                feature_type, strategy, confidence = "LANE_SURFACE_STREET", "mandatory", "high"
            elif element == "way" and key == "type" and value in WAY_TYPE_TO_SCENARIONET:
                feature_type, strategy, confidence = str(WAY_TYPE_TO_SCENARIONET[value]), "mandatory", "high"
            elif key == "subtype" and value == "crosswalk":
                feature_type, strategy, confidence = "CROSSWALK", "best-effort", "medium"
            writer.writerow(
                {
                    "element": element,
                    "tag_key": key,
                    "tag_value": value,
                    "cities": ";".join(sorted(cities)),
                    "scenarionet_feature_type": feature_type,
                    "confidence": confidence,
                    "v1_strategy": strategy,
                    "degrade_strategy": "omit feature and continue",
                }
            )
    return table_path


def _polyline_from_way(node_lookup: dict[str, np.ndarray], way_elem: ET.Element) -> np.ndarray:
    points = [node_lookup[node_ref.attrib["ref"]] for node_ref in way_elem.findall("nd") if node_ref.attrib["ref"] in node_lookup]
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(points, dtype=np.float32)


def _average_centerline(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    num_points = min(len(left), len(right))
    if num_points == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return (left[:num_points] + right[:num_points]) / 2.0


def _polygon_from_boundaries(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    num_points = min(len(left), len(right))
    if num_points == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate([left[:num_points], right[:num_points][::-1]], axis=0)


def parse_osm_map(osm_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    root = ET.parse(osm_path).getroot()
    node_lookup = {
        node.attrib["id"]: project_lon_lat_to_xy(float(node.attrib["lon"]), float(node.attrib["lat"]))
        for node in root.findall("node")
        if "lon" in node.attrib and "lat" in node.attrib
    }
    way_lookup = {way.attrib["id"]: way for way in root.findall("way")}
    way_polylines = {way_id: _polyline_from_way(node_lookup, way) for way_id, way in way_lookup.items()}

    map_features: dict[str, dict[str, Any]] = {}
    lane_centers: dict[str, np.ndarray] = {}
    for way_id, way_elem in way_lookup.items():
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way_elem.findall("tag")}
        feature_type = WAY_TYPE_TO_SCENARIONET.get(tags.get("type", ""))
        polyline = way_polylines[way_id]
        if feature_type is not None and len(polyline):
            map_features[f"boundary_{way_id}"] = {"type": feature_type, "polyline": polyline}

    for relation in root.findall("relation"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in relation.findall("tag")}
        if tags.get("type") != "lanelet":
            continue
        members = {member.attrib["role"]: member.attrib["ref"] for member in relation.findall("member")}
        left = way_polylines.get(members.get("left", ""))
        right = way_polylines.get(members.get("right", ""))
        if left is None or right is None or len(left) == 0 or len(right) == 0:
            continue
        feature_name = tags.get("name", relation.attrib["id"])
        polygon = _polygon_from_boundaries(left, right)
        if tags.get("subtype") == "crosswalk":
            map_features[f"crosswalk_{feature_name}"] = {"type": "CROSSWALK", "polygon": polygon}
            continue
        centerline = _average_centerline(left, right)
        lane_centers[feature_name] = centerline
        map_features[f"lane_{feature_name}"] = {
            "type": "LANE_SURFACE_STREET",
            "polyline": centerline,
            "polygon": polygon,
            "entry_lanes": [],
            "exit_lanes": [],
        }
    return map_features, lane_centers
