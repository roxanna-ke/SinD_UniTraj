from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIONET_ROOT = PROJECT_ROOT / "scenarionet"
for path in (PROJECT_ROOT, SCENARIONET_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sind_converter.data.discovery import discover_records
from sind_converter.lights.stopline_extraction import extract_stop_lines
from sind_converter.maps.osm import parse_osm_map


STATE_RED = 0
STATE_GREEN = 1
STATE_YELLOW = 3
STATE_UNKNOWN = -1
STATE_COLOR = {
    STATE_RED: "#d62828",
    STATE_GREEN: "#2a9d8f",
    STATE_YELLOW: "#f4a261",
    STATE_UNKNOWN: "#9aa0a6",
}
STATE_NAME = {
    STATE_RED: "Red",
    STATE_GREEN: "Green",
    STATE_YELLOW: "Yellow",
    STATE_UNKNOWN: "Unknown",
}
BASE_LANE_COLOR = "#d9ded9"
BASE_CONNECTOR_COLOR = "#b8c4b8"
BASE_STOPLINE_COLOR = "#62686f"


@dataclass(frozen=True)
class PhaseInterval:
    start_ms: float
    end_ms: float
    signature: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class CityPhaseSelection:
    city: str
    record_name: str
    cycle_start_ms: float
    cycle_end_ms: float
    phases: tuple[PhaseInterval, ...]


def _channel_sort_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)", str(name))
    number = int(match.group(1)) if match else 10_000
    return number, str(name)


def _channel_short_name(name: str) -> str:
    match = re.search(r"(\d+)", str(name))
    if match:
        return f"L{match.group(1)}"
    return str(name)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_artifacts(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_events = pd.read_csv(output_root / "traffic_light_channel_phase_events.csv")
    channel_cycle = pd.read_csv(output_root / "traffic_light_channel_cycle.csv")
    phase_events = phase_events[phase_events["channel_type"].astype(str).str.lower() == "vehicle"].copy()
    phase_events["time_ms"] = pd.to_numeric(phase_events["time_ms"], errors="coerce")
    phase_events["state"] = pd.to_numeric(phase_events["state"], errors="coerce").fillna(STATE_UNKNOWN).astype(int)
    channel_cycle["estimated_cycle_s"] = pd.to_numeric(channel_cycle["estimated_cycle_s"], errors="coerce")
    channel_cycle["num_cycles"] = pd.to_numeric(channel_cycle["num_cycles"], errors="coerce")
    return phase_events, channel_cycle


def _build_record_phase_intervals(record_phase: pd.DataFrame) -> list[PhaseInterval]:
    if record_phase.empty:
        return []
    record_phase = record_phase.sort_values(["traffic_light_channel", "time_ms"]).copy()
    channels = sorted(record_phase["traffic_light_channel"].unique(), key=_channel_sort_key)
    boundaries = np.sort(record_phase["time_ms"].dropna().unique())
    if len(boundaries) < 2:
        return []

    channel_times: dict[str, np.ndarray] = {}
    channel_states: dict[str, np.ndarray] = {}
    for channel in channels:
        channel_df = record_phase[record_phase["traffic_light_channel"] == channel].sort_values("time_ms")
        channel_times[channel] = channel_df["time_ms"].to_numpy(dtype=float)
        channel_states[channel] = channel_df["state"].to_numpy(dtype=int)

    intervals: list[PhaseInterval] = []
    current_start: float | None = None
    current_signature: tuple[tuple[str, int], ...] | None = None
    for start_ms, end_ms in zip(boundaries[:-1], boundaries[1:]):
        if end_ms <= start_ms:
            continue
        sample_ms = start_ms + 0.5 * (end_ms - start_ms)
        signature = []
        for channel in channels:
            idx = np.searchsorted(channel_times[channel], sample_ms, side="right") - 1
            state = int(channel_states[channel][idx]) if idx >= 0 else STATE_UNKNOWN
            signature.append((channel, state))
        frozen_signature = tuple(signature)
        if current_signature is None:
            current_start = float(start_ms)
            current_signature = frozen_signature
            continue
        if frozen_signature == current_signature:
            continue
        intervals.append(PhaseInterval(float(current_start), float(start_ms), current_signature))
        current_start = float(start_ms)
        current_signature = frozen_signature
    if current_signature is not None and current_start is not None:
        intervals.append(PhaseInterval(float(current_start), float(boundaries[-1]), current_signature))
    return [interval for interval in intervals if interval.end_ms > interval.start_ms]


def _reference_channel_cycles(record_phase: pd.DataFrame, record_cycle: pd.DataFrame) -> list[tuple[float, float]]:
    if record_phase.empty or record_cycle.empty:
        return []
    record_cycle = record_cycle.sort_values(
        ["num_cycles", "estimated_cycle_s", "traffic_light_channel"],
        ascending=[False, True, True],
    )
    ref_row = record_cycle.iloc[0]
    ref_channel = str(ref_row["traffic_light_channel"])
    channel_df = record_phase[record_phase["traffic_light_channel"] == ref_channel].sort_values("time_ms")
    green_starts = channel_df[channel_df["state"] == STATE_GREEN]["time_ms"].to_numpy(dtype=float)
    if len(green_starts) < 2:
        return []
    return [(float(start_ms), float(end_ms)) for start_ms, end_ms in zip(green_starts[:-1], green_starts[1:]) if end_ms > start_ms]


def _clip_cycle_phases(intervals: list[PhaseInterval], cycle_start_ms: float, cycle_end_ms: float) -> list[PhaseInterval]:
    clipped = []
    for interval in intervals:
        start_ms = max(interval.start_ms, cycle_start_ms)
        end_ms = min(interval.end_ms, cycle_end_ms)
        if end_ms <= start_ms:
            continue
        clipped.append(PhaseInterval(start_ms, end_ms, interval.signature))
    return clipped


def _pick_city_cycle(city: str, phase_events: pd.DataFrame, channel_cycle: pd.DataFrame) -> CityPhaseSelection:
    city_phase = phase_events[phase_events["city"] == city].copy()
    if city_phase.empty:
        raise ValueError(f"No vehicle phase events found for city={city}")
    candidates: list[tuple[int, float, str, list[PhaseInterval], tuple[float, float]]] = []
    for record_name in sorted(city_phase["record_name"].unique()):
        record_phase = city_phase[city_phase["record_name"] == record_name].copy()
        record_cycle = channel_cycle[(channel_cycle["city"] == city) & (channel_cycle["record_name"] == record_name)].copy()
        intervals = _build_record_phase_intervals(record_phase)
        cycles = _reference_channel_cycles(record_phase, record_cycle)
        for cycle_start_ms, cycle_end_ms in cycles:
            phases = _clip_cycle_phases(intervals, cycle_start_ms, cycle_end_ms)
            if len(phases) >= 2:
                candidates.append((len(phases), cycle_end_ms - cycle_start_ms, record_name, phases, (cycle_start_ms, cycle_end_ms)))
    if not candidates:
        raise ValueError(f"No valid cycle found for city={city}")
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    _, _, record_name, phases, (cycle_start_ms, cycle_end_ms) = candidates[0]
    return CityPhaseSelection(city, record_name, cycle_start_ms, cycle_end_ms, tuple(phases))


def _channel_state(signature: tuple[tuple[str, int], ...], channel_names: list[str]) -> int:
    lookup = {channel: state for channel, state in signature}
    states = [lookup.get(channel, STATE_UNKNOWN) for channel in channel_names]
    known = [state for state in states if state != STATE_UNKNOWN]
    if not known:
        return STATE_UNKNOWN
    if STATE_GREEN in known:
        return STATE_GREEN
    if STATE_YELLOW in known:
        return STATE_YELLOW
    if all(state == STATE_RED for state in known):
        return STATE_RED
    return known[0]


def _plot_city_binding_cycle(
    city: str,
    selection: CityPhaseSelection,
    map_features: dict[str, dict[str, Any]],
    stop_lines_df: pd.DataFrame,
    channel_groups: list[dict[str, Any]],
    lane_bindings: list[dict[str, Any]],
    output_path: Path,
) -> None:
    lane_binding_lookup = {(str(row["stopline_id"]), str(row["movement"])): row for row in lane_bindings}
    active_groups_per_phase = []
    all_points = []
    for feature_id, feature in map_features.items():
        geometry = feature.get("polyline", feature.get("polygon"))
        if geometry is None:
            continue
        pts = np.asarray(geometry, dtype=float)
        if pts.ndim == 2 and pts.shape[0] > 0:
            all_points.append(pts[:, :2])
    all_points.append(stop_lines_df[["x1", "y1"]].to_numpy(dtype=float))
    all_points.append(stop_lines_df[["x2", "y2"]].to_numpy(dtype=float))
    stacked = np.vstack(all_points)
    x_min, y_min = stacked.min(axis=0)
    x_max, y_max = stacked.max(axis=0)
    margin = max(x_max - x_min, y_max - y_min, 1.0) * 0.1

    num_phases = len(selection.phases)
    ncols = min(3, num_phases)
    nrows = int(math.ceil(num_phases / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.8 * ncols, 6.4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for phase_index, phase in enumerate(selection.phases):
        ax = axes_flat[phase_index]
        _draw_base_map(ax, map_features, stop_lines_df)
        phase_rows = []
        for group in channel_groups:
            state = _channel_state(phase.signature, list(group["traffic_light_channels"]))
            group_color = STATE_COLOR[state]
            for movement_row in group["movements"]:
                key = (str(movement_row["stopline_id"]), str(movement_row["movement"]))
                lane_row = lane_binding_lookup.get(key)
                if lane_row is None:
                    continue
                phase_rows.append((group, movement_row, lane_row, state, group_color))
                _draw_lane_binding(ax, map_features, stop_lines_df, lane_row, group, state, group_color)

        visible_rows = [row for row in phase_rows if row[3] in {STATE_GREEN, STATE_YELLOW}]
        rows_to_show = visible_rows if visible_rows else phase_rows
        for row_index, (group, movement_row, lane_row, state, _) in enumerate(rows_to_show[:8]):
            stopline_id = str(movement_row["stopline_id"])
            label = f"{_channel_short_name(group['traffic_light_channels'][0])} | stopline {stopline_id} | {movement_row['movement']} | {STATE_NAME[state]}"
            y = 0.98 - row_index * 0.05
            ax.text(
                0.02,
                y,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            )
        if len(rows_to_show) > 8:
            ax.text(
                0.02,
                0.98 - 8 * 0.05,
                f"... {len(rows_to_show) - 8} more controlled movements",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#5f6368",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.2},
            )
        duration_s = (phase.end_ms - phase.start_ms) / 1000.0
        phase_state_summary = " | ".join(f"{_channel_short_name(channel)}={STATE_NAME[state][0]}" for channel, state in phase.signature)
        ax.set_title(f"{city} | {selection.record_name}\nPhase {phase_index + 1} ({duration_s:.1f}s): {phase_state_summary}", fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        active_groups_per_phase.append(phase_rows)

    for ax in axes_flat[num_phases:]:
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _draw_base_map(ax: plt.Axes, map_features: dict[str, dict[str, Any]], stop_lines_df: pd.DataFrame) -> None:
    for feature_id, feature in map_features.items():
        if not str(feature_id).startswith("lane_"):
            continue
        polyline = np.asarray(feature.get("polyline", np.zeros((0, 3), dtype=np.float32)), dtype=float)
        if polyline.ndim != 2 or polyline.shape[0] == 0:
            continue
        color = BASE_CONNECTOR_COLOR if "_to_" in str(feature_id) else BASE_LANE_COLOR
        linewidth = 2.8 if "_to_" in str(feature_id) else 1.8
        ax.plot(polyline[:, 0], polyline[:, 1], color=color, linewidth=linewidth, alpha=0.8, solid_capstyle="round", zorder=1)
    for line in stop_lines_df.itertuples(index=False):
        ax.plot([line.x1, line.x2], [line.y1, line.y2], color=BASE_STOPLINE_COLOR, linewidth=3.0, alpha=0.85, zorder=3)
        ax.text(
            float(line.mid_x),
            float(line.mid_y),
            str(line.stopline_id),
            fontsize=8,
            ha="center",
            va="center",
            color="#40464d",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.8},
            zorder=4,
        )


def _draw_lane_binding(
    ax: plt.Axes,
    map_features: dict[str, dict[str, Any]],
    stop_lines_df: pd.DataFrame,
    lane_row: dict[str, Any],
    group: dict[str, Any],
    state: int,
    color: str,
) -> None:
    linewidth = 5.0 if state == STATE_GREEN else 4.0 if state == STATE_YELLOW else 3.2
    alpha = 0.95 if state in {STATE_GREEN, STATE_YELLOW} else 0.6
    for lane_id in lane_row["lane_ids"]:
        feature = map_features.get(lane_id)
        if feature is None:
            continue
        polyline = np.asarray(feature.get("polyline", np.zeros((0, 3), dtype=np.float32)), dtype=float)
        if polyline.ndim != 2 or polyline.shape[0] == 0:
            continue
        ax.plot(polyline[:, 0], polyline[:, 1], color=color, linewidth=linewidth, alpha=alpha, solid_capstyle="round", zorder=5)

    stopline_df = stop_lines_df[stop_lines_df["stopline_id"].astype(str) == str(lane_row["stopline_id"])]
    if not stopline_df.empty:
        row = stopline_df.iloc[0]
        ax.plot([row["x1"], row["x2"]], [row["y1"], row["y2"]], color=color, linewidth=5.0, alpha=alpha, zorder=6)
        label = str(group["traffic_light_channels"][0])
        ax.text(
            float(row["mid_x"]),
            float(row["mid_y"]) + 1.5,
            _channel_short_name(label),
            fontsize=8.5,
            ha="center",
            va="bottom",
            color=color,
            bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.8, "pad": 1.0},
            zorder=7,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one signal cycle per city with active lanes, stoplines, and channel labels.")
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "SinD" / "Dataset")
    parser.add_argument("--map-fallback-root", type=Path, default=PROJECT_ROOT / "SinD" / "Data")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output" / "lights")
    parser.add_argument("--binding-root", type=Path, default=PROJECT_ROOT / "sind_converter" / "lights" / "config")
    parser.add_argument("--figure-root", type=Path, default=PROJECT_ROOT / "output" / "signal_binding_visualizations")
    parser.add_argument("--cities", nargs="*", default=["Xi_an", "Changchun", "Chongqing", "Tianjin"])
    args = parser.parse_args()

    phase_events, channel_cycle = _load_artifacts(args.output_root)
    records = discover_records(args.data_root, args.map_fallback_root, cities=args.cities)
    record_by_city = {}
    for record in records:
        record_by_city.setdefault(record.city, record)

    for city in args.cities:
        record = record_by_city.get(city)
        if record is None:
            print(f"[skip] no record discovered for city={city}")
            continue
        channel_groups = _load_json(args.binding_root / "channel_groups" / f"{city}.json")["channel_groups"]
        lane_bindings = _load_json(args.binding_root / "lane_bindings" / f"{city}.json")["lane_bindings"]
        if not channel_groups or not lane_bindings:
            print(f"[skip] empty bindings for city={city}")
            continue
        selection = _pick_city_cycle(city, phase_events, channel_cycle)
        map_features, _ = parse_osm_map(record.map_path)
        stop_lines = extract_stop_lines(record.map_path, city)
        stop_lines_df = pd.DataFrame([line.as_row() for line in stop_lines])
        output_path = args.figure_root / f"{city.lower()}_binding_cycle.png"
        _plot_city_binding_cycle(city, selection, map_features, stop_lines_df, channel_groups, lane_bindings, output_path)
        print(f"[ok] {city}: {selection.record_name} phases={len(selection.phases)} -> {output_path}")


if __name__ == "__main__":
    main()
