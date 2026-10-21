from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sind_converter.data.discovery import RecordDescription
from sind_converter.lights.stopline_extraction import StopLine


@dataclass(frozen=True)
class ChannelGroup:
    group_id: str
    traffic_light_channels: tuple[str, ...]
    movements: tuple[tuple[str, str], ...]
    record_names: tuple[str, ...]
    confidence: str
    source: str


@dataclass(frozen=True)
class LaneBinding:
    stopline_id: str
    movement: str
    lane_feature_ids: tuple[str, ...]
    stop_point: tuple[float, float, float] | None
    record_names: tuple[str, ...]
    confidence: str
    source: str


@dataclass(frozen=True)
class LaneSignalBinding:
    lane_feature_id: str
    stopline_id: str
    movement: str
    traffic_light_channels: tuple[str, ...]
    stop_point: tuple[float, float, float]
    binding_group: str
    confidence: str
    source: str


def resolve_signal_bindings(
    record: RecordDescription,
    traffic_light: pd.DataFrame | None,
    map_features: dict[str, dict[str, Any]],
    stop_lines: list[StopLine],
    binding_root: Path | None,
) -> tuple[LaneSignalBinding, ...]:
    if binding_root is None or traffic_light is None or traffic_light.empty:
        return ()

    binding_root = Path(binding_root)
    channel_groups = _load_channel_groups(record.city, binding_root)
    lane_bindings = _load_lane_bindings(record.city, binding_root)
    if not channel_groups or not lane_bindings:
        return ()

    _validate_channel_columns(record, traffic_light, channel_groups)
    stopline_lookup = {line.stopline_id: line for line in stop_lines}
    lane_feature_lookup = {key: value for key, value in map_features.items() if str(key).startswith("lane_")}

    runtime_bindings: list[LaneSignalBinding] = []
    assigned_lanes: dict[str, tuple[str, str]] = {}
    lane_binding_lookup = {(binding.stopline_id, binding.movement): binding for binding in lane_bindings}
    for group in channel_groups:
        if group.record_names and record.record_name not in group.record_names:
            continue
        for stopline_id, movement in group.movements:
            binding = lane_binding_lookup.get((stopline_id, movement))
            if binding is None or (binding.record_names and record.record_name not in binding.record_names):
                continue
            stop_line = stopline_lookup.get(stopline_id)
            if stop_line is None:
                raise ValueError(f"{record.city} binding references unknown stopline_id={stopline_id}")
            for lane_feature_id in binding.lane_feature_ids:
                if lane_feature_id not in lane_feature_lookup:
                    raise ValueError(f"{record.city} binding references unknown lane feature {lane_feature_id}")
                prior = assigned_lanes.get(lane_feature_id)
                if prior is not None and prior != (stopline_id, movement):
                    raise ValueError(
                        f"{record.city} lane {lane_feature_id} is assigned to multiple movement bindings: "
                        f"{prior} and {(stopline_id, movement)}"
                    )
                assigned_lanes[lane_feature_id] = (stopline_id, movement)
                stop_point = binding.stop_point or _derive_stop_point(lane_feature_lookup[lane_feature_id], stop_line)
                runtime_bindings.append(
                    LaneSignalBinding(
                        lane_feature_id=lane_feature_id,
                        stopline_id=stopline_id,
                        movement=movement,
                        traffic_light_channels=group.traffic_light_channels,
                        stop_point=stop_point,
                        binding_group=group.group_id,
                        confidence=_merge_confidence(group.confidence, binding.confidence),
                        source=_merge_source(group.source, binding.source),
                    )
                )
    return tuple(runtime_bindings)


def _load_channel_groups(city: str, binding_root: Path) -> tuple[ChannelGroup, ...]:
    path = _resolve_binding_path(binding_root / "channel_groups", city)
    if path is None:
        return ()
    payload = _load_binding_payload(path)
    rows = payload.get("channel_groups", []) if isinstance(payload, dict) else []
    groups: list[ChannelGroup] = []
    for idx, row in enumerate(rows):
        group_id = str(row.get("group_id", f"{city.lower()}_{idx}"))
        channels = tuple(str(x) for x in row.get("traffic_light_channels", []))
        if not channels:
            raise ValueError(f"{path} group {group_id} is missing traffic_light_channels")
        movements = tuple((str(m["stopline_id"]), str(m["movement"]).strip().lower()) for m in row.get("movements", []))
        if not movements:
            raise ValueError(f"{path} group {group_id} is missing movements")
        groups.append(
            ChannelGroup(
                group_id=group_id,
                traffic_light_channels=channels,
                movements=movements,
                record_names=tuple(str(x) for x in row.get("record_names", [])),
                confidence=str(row.get("confidence", "manual")).strip().lower(),
                source=str(row.get("source", "declarative")).strip(),
            )
        )
    return tuple(groups)


def _load_lane_bindings(city: str, binding_root: Path) -> tuple[LaneBinding, ...]:
    path = _resolve_binding_path(binding_root / "lane_bindings", city)
    if path is None:
        return ()
    payload = _load_binding_payload(path)
    rows = payload.get("lane_bindings", []) if isinstance(payload, dict) else []
    bindings: list[LaneBinding] = []
    for row in rows:
        lane_ids = tuple(_normalize_lane_feature_id(str(x)) for x in row.get("lane_ids", []))
        if not lane_ids:
            raise ValueError(f"{path} binding for stopline={row.get('stopline_id')} movement={row.get('movement')} has no lane_ids")
        stop_point = row.get("stop_point")
        bindings.append(
            LaneBinding(
                stopline_id=str(row["stopline_id"]),
                movement=str(row["movement"]).strip().lower(),
                lane_feature_ids=lane_ids,
                stop_point=_coerce_stop_point(stop_point),
                record_names=tuple(str(x) for x in row.get("record_names", [])),
                confidence=str(row.get("confidence", "manual")).strip().lower(),
                source=str(row.get("source", "declarative")).strip(),
            )
        )
    return tuple(bindings)


def _validate_channel_columns(record: RecordDescription, traffic_light: pd.DataFrame, channel_groups: tuple[ChannelGroup, ...]) -> None:
    available = {str(col) for col in traffic_light.columns if "traffic" in str(col).lower() and "light" in str(col).lower()}
    for group in channel_groups:
        for channel in group.traffic_light_channels:
            if channel not in available:
                raise ValueError(
                    f"{record.city}/{record.record_name} binding references unknown traffic-light column {channel!r}; "
                    f"available={sorted(available)}"
                )


def _normalize_lane_feature_id(value: str) -> str:
    return value if value.startswith("lane_") else f"lane_{value}"


def _load_binding_payload(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} must be JSON-compatible YAML so it can be loaded without optional YAML dependencies"
        ) from exc


def _resolve_binding_path(root: Path, city: str) -> Path | None:
    for suffix in (".json", ".yaml"):
        candidate = root / f"{city}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _coerce_stop_point(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) not in {2, 3}:
        raise ValueError(f"stop_point must be a 2D or 3D coordinate, got {value!r}")
    coords = [float(x) for x in value]
    if len(coords) == 2:
        coords.append(0.0)
    return tuple(coords)


def _derive_stop_point(lane_feature: dict[str, Any], stop_line: StopLine) -> tuple[float, float, float]:
    polyline = np.asarray(lane_feature.get("polyline", np.zeros((0, 3), dtype=np.float32)), dtype=np.float64)
    if polyline.ndim != 2 or polyline.shape[0] == 0:
        return (float(stop_line.mid_x), float(stop_line.mid_y), 0.0)
    points = polyline[:, :2]
    distances = np.linalg.norm(points - stop_line.midpoint[None, :], axis=1)
    idx = int(np.argmin(distances))
    stop_point = polyline[idx]
    if stop_point.shape[0] == 2:
        return (float(stop_point[0]), float(stop_point[1]), 0.0)
    return (float(stop_point[0]), float(stop_point[1]), float(stop_point[2]))


def _merge_confidence(group_confidence: str, lane_confidence: str) -> str:
    levels = {"low": 0, "medium": 1, "high": 2, "manual": 2}
    reverse = {0: "low", 1: "medium", 2: "high"}
    merged = min(levels.get(group_confidence, 1), levels.get(lane_confidence, 1))
    return reverse.get(merged, "medium")


def _merge_source(group_source: str, lane_source: str) -> str:
    if group_source == lane_source:
        return group_source
    return f"{group_source}+{lane_source}"
