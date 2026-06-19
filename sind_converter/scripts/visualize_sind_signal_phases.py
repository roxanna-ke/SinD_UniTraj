from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sind_converter.data.discovery import RecordDescription, discover_records


STATE_LABELS = {
    0: "R",
    1: "G",
    3: "Y",
    -1: "U",
}


@dataclass(frozen=True)
class PhaseInterval:
    start_ms: float
    end_ms: float
    signature: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class CityCycleSelection:
    city: str
    record_name: str
    cycle_index: int
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


def _load_vehicle_tracks(record: RecordDescription) -> pd.DataFrame:
    df = pd.read_csv(record.vehicle_tracks_path)
    required = {"track_id", "timestamp_ms", "x", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{record.vehicle_tracks_path} missing columns: {sorted(missing)}")
    for col in ("timestamp_ms", "x", "y"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[np.isfinite(df[["timestamp_ms", "x", "y"]].to_numpy(dtype=float)).all(axis=1)].copy()
    df["track_id"] = df["track_id"].astype(str)
    return df.sort_values(["track_id", "timestamp_ms"])


def _load_artifacts(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phase_events = pd.read_csv(output_root / "traffic_light_channel_phase_events.csv")
    stop_lines = pd.read_csv(output_root / "stop_lines.csv")
    channel_cycle = pd.read_csv(output_root / "traffic_light_channel_cycle.csv")
    phase_events = phase_events[phase_events["channel_type"].astype(str).str.lower() == "vehicle"].copy()
    phase_events["time_ms"] = pd.to_numeric(phase_events["time_ms"], errors="coerce")
    phase_events["state"] = pd.to_numeric(phase_events["state"], errors="coerce").fillna(-1).astype(int)
    channel_cycle["estimated_cycle_s"] = pd.to_numeric(channel_cycle["estimated_cycle_s"], errors="coerce")
    channel_cycle["num_cycles"] = pd.to_numeric(channel_cycle["num_cycles"], errors="coerce")
    return phase_events, stop_lines, channel_cycle


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
        signature_list: list[tuple[str, int]] = []
        for channel in channels:
            idx = np.searchsorted(channel_times[channel], sample_ms, side="right") - 1
            state = int(channel_states[channel][idx]) if idx >= 0 else -1
            signature_list.append((channel, state))
        signature = tuple(signature_list)

        if current_signature is None:
            current_start = float(start_ms)
            current_signature = signature
            continue
        if signature == current_signature:
            continue
        intervals.append(PhaseInterval(start_ms=float(current_start), end_ms=float(start_ms), signature=current_signature))
        current_start = float(start_ms)
        current_signature = signature

    if current_signature is not None and current_start is not None:
        intervals.append(PhaseInterval(start_ms=float(current_start), end_ms=float(boundaries[-1]), signature=current_signature))
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
    green_starts = channel_df[channel_df["state"] == 1]["time_ms"].to_numpy(dtype=float)
    if len(green_starts) < 2:
        return []
    cycles: list[tuple[float, float]] = []
    for start_ms, end_ms in zip(green_starts[:-1], green_starts[1:]):
        if end_ms > start_ms:
            cycles.append((float(start_ms), float(end_ms)))
    return cycles


def _clip_cycle_phases(intervals: list[PhaseInterval], cycle_start_ms: float, cycle_end_ms: float) -> list[PhaseInterval]:
    clipped: list[PhaseInterval] = []
    for interval in intervals:
        start_ms = max(interval.start_ms, cycle_start_ms)
        end_ms = min(interval.end_ms, cycle_end_ms)
        if end_ms <= start_ms:
            continue
        clipped.append(PhaseInterval(start_ms=start_ms, end_ms=end_ms, signature=interval.signature))
    return clipped


def _phase_track_segments(vehicle_df: pd.DataFrame, phase: PhaseInterval) -> list[np.ndarray]:
    phase_df = vehicle_df[(vehicle_df["timestamp_ms"] >= phase.start_ms) & (vehicle_df["timestamp_ms"] < phase.end_ms)].copy()
    if phase_df.empty:
        return []
    segments: list[np.ndarray] = []
    for _, track in phase_df.groupby("track_id", sort=False):
        points = track[["x", "y"]].to_numpy(dtype=float)
        if len(points) >= 2:
            segments.append(points)
    return segments


def _phase_label(signature: tuple[tuple[str, int], ...]) -> str:
    parts = [f"{_channel_short_name(channel)}={STATE_LABELS.get(int(state), 'U')}" for channel, state in signature]
    return " | ".join(parts)


def _pick_city_cycle(
    city: str,
    record_lookup: dict[tuple[str, str], RecordDescription],
    phase_events: pd.DataFrame,
    channel_cycle: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[CityCycleSelection, pd.DataFrame]:
    city_records = sorted(phase_events[phase_events["city"] == city]["record_name"].unique())
    if not city_records:
        raise ValueError(f"No phase events for city={city}")

    candidate_records = list(city_records)
    rng.shuffle(candidate_records)

    valid_candidates: list[tuple[CityCycleSelection, pd.DataFrame]] = []
    for record_name in candidate_records:
        record = record_lookup[(city, record_name)]
        vehicle_df = _load_vehicle_tracks(record)
        record_phase = phase_events[(phase_events["city"] == city) & (phase_events["record_name"] == record_name)].copy()
        record_cycle = channel_cycle[(channel_cycle["city"] == city) & (channel_cycle["record_name"] == record_name)].copy()
        intervals = _build_record_phase_intervals(record_phase)
        cycles = _reference_channel_cycles(record_phase, record_cycle)
        if not intervals or not cycles:
            continue

        cycle_indices = list(range(len(cycles)))
        rng.shuffle(cycle_indices)
        for cycle_index in cycle_indices:
            cycle_start_ms, cycle_end_ms = cycles[cycle_index]
            phases = _clip_cycle_phases(intervals, cycle_start_ms, cycle_end_ms)
            if len(phases) < 2:
                continue
            if any(len(_phase_track_segments(vehicle_df, phase)) == 0 for phase in phases):
                continue
            selection = CityCycleSelection(
                city=city,
                record_name=record_name,
                cycle_index=cycle_index,
                cycle_start_ms=cycle_start_ms,
                cycle_end_ms=cycle_end_ms,
                phases=tuple(phases),
            )
            valid_candidates.append((selection, vehicle_df))

    if not valid_candidates:
        raise ValueError(f"No valid cycle with non-empty trajectories for every phase in city={city}")
    return valid_candidates[int(rng.integers(len(valid_candidates)))]


def _plot_city_cycle(
    selection: CityCycleSelection,
    vehicle_df: pd.DataFrame,
    stop_lines: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    phase_segments = [_phase_track_segments(vehicle_df, phase) for phase in selection.phases]
    all_points = []
    for segments in phase_segments:
        for seg in segments:
            all_points.append(seg)
    stop_points = [stop_lines[["x1", "y1"]].to_numpy(dtype=float), stop_lines[["x2", "y2"]].to_numpy(dtype=float)]
    if all_points:
        stacked = np.vstack([*all_points, *stop_points])
    else:
        stacked = np.vstack(stop_points)
    x_min, y_min = stacked.min(axis=0)
    x_max, y_max = stacked.max(axis=0)
    margin = max(x_max - x_min, y_max - y_min, 1.0) * 0.08

    num_phases = len(selection.phases)
    ncols = min(3, num_phases)
    nrows = int(math.ceil(num_phases / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.5 * ncols, 6.0 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (phase, segments) in enumerate(zip(selection.phases, phase_segments)):
        ax = axes_flat[idx]
        for line in stop_lines.itertuples(index=False):
            ax.plot([line.x1, line.x2], [line.y1, line.y2], color="crimson", linewidth=3.0, solid_capstyle="round", zorder=3)
        for seg in segments:
            ax.plot(seg[:, 0], seg[:, 1], color="steelblue", alpha=0.45, linewidth=1.0, zorder=2)
        duration_s = (phase.end_ms - phase.start_ms) / 1000.0
        ax.set_title(f"Phase {idx + 1}: {_phase_label(phase.signature)}\n{duration_s:.1f}s", fontsize=11)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    for ax in axes_flat[num_phases:]:
        ax.axis("off")

    fig.suptitle(
        f"{selection.city} | {selection.record_name} | cycle {selection.cycle_index} | "
        f"{(selection.cycle_end_ms - selection.cycle_start_ms) / 1000.0:.1f}s",
        fontsize=14,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "city": selection.city,
        "record_name": selection.record_name,
        "cycle_index": selection.cycle_index,
        "cycle_start_ms": round(selection.cycle_start_ms, 3),
        "cycle_end_ms": round(selection.cycle_end_ms, 3),
        "num_phases": num_phases,
        "figure_path": str(output_path),
        "phase_labels": " || ".join(_phase_label(phase.signature) for phase in selection.phases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize one valid traffic-light cycle per SinD city.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--map-fallback-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True, help="Directory containing traffic-light inference CSVs.")
    parser.add_argument("--fig-root", type=Path, default=None, help="Output directory for city figures.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_root = args.output_root
    fig_root = args.fig_root or (output_root / "visualizations")
    phase_events, stop_lines, channel_cycle = _load_artifacts(output_root)

    map_fallback_root = args.map_fallback_root or args.data_root
    records = discover_records(args.data_root, map_fallback_root)
    record_lookup = {(record.city, record.record_name): record for record in records}

    rng = np.random.default_rng(args.seed)
    summary_rows: list[dict[str, Any]] = []
    for city in sorted(stop_lines["city"].unique()):
        selection, vehicle_df = _pick_city_cycle(city, record_lookup, phase_events, channel_cycle, rng)
        city_stop_lines = stop_lines[stop_lines["city"] == city].copy()
        output_path = fig_root / f"{city.lower()}_phase_cycle.png"
        summary = _plot_city_cycle(selection, vehicle_df, city_stop_lines, output_path)
        summary_rows.append(summary)
        print(
            f"[done] {city}: record={selection.record_name} cycle={selection.cycle_index} "
            f"phases={len(selection.phases)} -> {output_path}"
        )

    pd.DataFrame(summary_rows).to_csv(fig_root / "phase_cycle_summary.csv", index=False)
    print(f"[done] wrote figures to {fig_root}")


if __name__ == "__main__":
    main()
