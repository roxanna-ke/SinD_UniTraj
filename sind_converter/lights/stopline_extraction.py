from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sind_converter.maps.osm import project_lon_lat_to_xy


@dataclass(frozen=True)
class StopLine:
    city: str
    stopline_id: str
    osm_way_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    mid_x: float
    mid_y: float
    nx: float
    ny: float

    @property
    def p1(self) -> np.ndarray:
        return np.array([self.x1, self.y1], dtype=np.float64)

    @property
    def p2(self) -> np.ndarray:
        return np.array([self.x2, self.y2], dtype=np.float64)

    @property
    def midpoint(self) -> np.ndarray:
        return np.array([self.mid_x, self.mid_y], dtype=np.float64)

    @property
    def normal(self) -> np.ndarray:
        return np.array([self.nx, self.ny], dtype=np.float64)

    def as_row(self) -> dict[str, Any]:
        return {
            "city": self.city,
            "stopline_id": self.stopline_id,
            "osm_way_id": self.osm_way_id,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "mid_x": self.mid_x,
            "mid_y": self.mid_y,
            "nx": self.nx,
            "ny": self.ny,
        }


def _element_tags(elem: ET.Element) -> dict[str, str]:
    return {tag.attrib.get("k", ""): tag.attrib.get("v", "") for tag in elem.findall("tag")}


def _node_lookup(root: ET.Element) -> dict[str, np.ndarray]:
    return {
        node.attrib["id"]: project_lon_lat_to_xy(float(node.attrib["lon"]), float(node.attrib["lat"]))[:2].astype(np.float64)
        for node in root.findall("node")
        if "id" in node.attrib and "lon" in node.attrib and "lat" in node.attrib
    }


def _way_points(node_lookup: dict[str, np.ndarray], way: ET.Element) -> np.ndarray:
    points = [node_lookup[nd.attrib["ref"]] for nd in way.findall("nd") if nd.attrib.get("ref") in node_lookup]
    if not points:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def _directed_normal(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    direction = p2 - p1
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return np.array([0.0, 1.0], dtype=np.float64)
    normal = np.array([-direction[1], direction[0]], dtype=np.float64) / norm
    return normal


def extract_stop_lines(osm_path: Path, city: str) -> list[StopLine]:
    root = ET.parse(osm_path).getroot()
    nodes = _node_lookup(root)
    stop_lines: list[StopLine] = []
    for way in root.findall("way"):
        tags = _element_tags(way)
        if tags.get("type") != "stop_line":
            continue
        points = _way_points(nodes, way)
        if len(points) < 2:
            continue
        p1 = points[0]
        p2 = points[-1]
        midpoint = (p1 + p2) / 2.0
        normal = _directed_normal(p1, p2)
        way_id = way.attrib["id"]
        stop_lines.append(
            StopLine(
                city=city,
                stopline_id=way_id,
                osm_way_id=way_id,
                x1=float(p1[0]),
                y1=float(p1[1]),
                x2=float(p2[0]),
                y2=float(p2[1]),
                mid_x=float(midpoint[0]),
                mid_y=float(midpoint[1]),
                nx=float(normal[0]),
                ny=float(normal[1]),
            )
        )
    return sorted(stop_lines, key=lambda line: line.stopline_id)


def _traffic_light_channel_name(way: ET.Element | None) -> str:
    if way is None:
        return "unknown"
    tags = _element_tags(way)
    for key in ("name", "ref", "id"):
        value = tags.get(key)
        if not value:
            continue
        match = re.search(r"traffic\s*light\s*\d+", value, flags=re.IGNORECASE)
        if match:
            text = match.group(0)
            return re.sub(r"\s+", " ", text).strip()
    return "unknown"


def extract_traffic_light_relations(osm_path: Path, city: str) -> list[dict[str, str]]:
    root = ET.parse(osm_path).getroot()
    way_lookup = {way.attrib["id"]: way for way in root.findall("way") if "id" in way.attrib}
    rows: list[dict[str, str]] = []
    for relation in root.findall("relation"):
        tags = _element_tags(relation)
        if tags.get("type") != "regulatory_element" or tags.get("subtype") != "traffic_light":
            continue
        ref_lines = [member.attrib.get("ref", "") for member in relation.findall("member") if member.attrib.get("role") == "ref_line"]
        refers = [member.attrib.get("ref", "") for member in relation.findall("member") if member.attrib.get("role") == "refers"]
        for ref_line in ref_lines or [""]:
            for refers_way in refers or [""]:
                rows.append(
                    {
                        "city": city,
                        "osm_relation_id": relation.attrib.get("id", ""),
                        "osm_relation_expected_stopline": ref_line or "unknown",
                        "osm_relation_refers_way": refers_way or "unknown",
                        "osm_relation_expected_channel": _traffic_light_channel_name(way_lookup.get(refers_way)),
                    }
                )
    return rows


STOP_LINE_FIELDS = ["city", "stopline_id", "osm_way_id", "x1", "y1", "x2", "y2", "mid_x", "mid_y", "nx", "ny"]

