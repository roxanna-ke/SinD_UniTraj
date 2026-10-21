from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sind_converter.data.discovery import RecordDescription, discover_records, normalize_city_name, resolve_map_path
from sind_converter.data.loading import load_record
from sind_converter.maps.osm import parse_osm_map


TARGET_TYPES = {"car", "truck", "bus"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a SinD city OSM intersection map and optionally overlay predicted vs. actual trajectories."
    )
    parser.add_argument("--city", required=True, help="SinD city name: Tianjin, Changchun, Chongqing, Xi_an.")
    parser.add_argument("--record", help="Record directory name. If omitted, only the city map is drawn unless --track-id is used.")
    parser.add_argument("--track-id", help="Track id to draw from the record. Defaults to the longest car/truck/bus track.")
    parser.add_argument("--frame-id", type=int, help="Current frame. Defaults to the first frame with enough future for the track.")
    parser.add_argument("--past-len", type=int, default=21)
    parser.add_argument("--future-len", type=int, default=60)
    parser.add_argument("--prediction-path", type=Path, help="CSV or NPZ containing predicted xy points.")
    parser.add_argument("--prediction-key", default=None, help="NPZ key for prediction points. Defaults to pred/prediction/arr_0.")
    parser.add_argument("--data-root", type=Path, default=Path("SinD/Dataset"))
    parser.add_argument("--map-fallback-root", type=Path, default=Path("SinD/Data"))
    parser.add_argument("--output", type=Path, default=Path("output/osm_prediction_visualizations/osm_prediction.png"))
    parser.add_argument("--title", default=None)
    parser.add_argument("--show-context", action="store_true", help="Draw other tracks present at --frame-id.")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def _feature_geometry(feature: dict[str, Any]) -> tuple[np.ndarray, bool]:
    if "polygon" in feature:
        points = np.asarray(feature["polygon"], dtype=np.float32)
        closed = True
    elif "polyline" in feature:
        points = np.asarray(feature["polyline"], dtype=np.float32)
        closed = False
    else:
        return np.zeros((0, 2), dtype=np.float32), False
    if points.ndim != 2 or len(points) == 0:
        return np.zeros((0, 2), dtype=np.float32), closed
    return points[:, :2], closed


def _draw_osm_map(ax: plt.Axes, map_features: dict[str, dict[str, Any]]) -> None:
    for feature in map_features.values():
        points, closed = _feature_geometry(feature)
        if len(points) < 2:
            continue
        feature_type = str(feature.get("type", ""))
        if "BOUNDARY" in feature_type:
            style = {"color": "#111111", "linewidth": 1.2, "linestyle": "-"}
        elif "BROKEN" in feature_type:
            style = {"color": "#b8b8b8", "linewidth": 0.8, "linestyle": (0, (4, 4))}
        elif "ROAD_LINE" in feature_type:
            style = {"color": "#b0b0b0", "linewidth": 0.9, "linestyle": "-"}
        elif "LANE" in feature_type:
            style = {"color": "#9b9b9b", "linewidth": 0.55, "linestyle": ":"}
        elif feature_type == "CROSSWALK":
            style = {"color": "#c0c0c0", "linewidth": 0.8, "linestyle": "-"}
        else:
            style = {"color": "#c7c7c7", "linewidth": 0.6, "linestyle": "-"}
        plot_points = points
        if closed and len(points) >= 3:
            plot_points = np.vstack([points, points[:1]])
        ax.plot(plot_points[:, 0], plot_points[:, 1], alpha=0.95, zorder=3, **style)


def _find_record(data_root: Path, map_fallback_root: Path, city: str, record_name: str | None) -> RecordDescription | None:
    records = discover_records(data_root, map_fallback_root, cities=[normalize_city_name(city)])
    if record_name is None:
        return records[0] if records else None
    for record in records:
        if record.record_name == record_name:
            return record
    available = ", ".join(record.record_name for record in records[:10])
    raise FileNotFoundError(f"No record {record_name!r} for city={city!r}. First available records: {available}")


def _combined_tracks(record: RecordDescription) -> pd.DataFrame:
    loaded = load_record(record)
    veh = loaded.vehicle_tracks.copy()
    ped = loaded.pedestrian_tracks.copy()
    if "agent_type" not in ped.columns:
        ped["agent_type"] = "pedestrian"
    return pd.concat([veh, ped], ignore_index=True, sort=False)


def _choose_track_id(tracks: pd.DataFrame) -> str:
    candidates = tracks[tracks["agent_type"].astype(str).str.lower().isin(TARGET_TYPES)]
    if candidates.empty:
        candidates = tracks
    counts = candidates.groupby("track_id", sort=True)["frame_id"].nunique().sort_values(ascending=False)
    if counts.empty:
        raise ValueError("No drawable tracks found in the selected record.")
    return str(counts.index[0])


def _choose_frame_id(track: pd.DataFrame, past_len: int, future_len: int, requested: int | None) -> int:
    frames = np.sort(track["frame_id"].astype(int).unique())
    if len(frames) == 0:
        raise ValueError("Selected track has no frames.")
    if requested is not None:
        return int(requested)
    if len(frames) >= past_len + future_len:
        return int(frames[past_len - 1])
    return int(frames[len(frames) // 2])


def _track_window_points(track: pd.DataFrame, frame_id: int, past_len: int, future_len: int) -> tuple[np.ndarray, np.ndarray]:
    track = track.sort_values("frame_id")
    past = track[(track["frame_id"] <= frame_id)].tail(past_len)
    future = track[(track["frame_id"] > frame_id)].head(future_len)
    return past[["x", "y"]].to_numpy(dtype=np.float32), future[["x", "y"]].to_numpy(dtype=np.float32)


def _load_prediction(path: Path, key: str | None = None) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        candidate_keys = [key] if key else ["pred", "prediction", "predicted_trajectory", "arr_0"]
        for candidate in candidate_keys:
            if candidate and candidate in data:
                arr = np.asarray(data[candidate], dtype=np.float32)
                break
        else:
            raise KeyError(f"No prediction key found in {path}; tried {candidate_keys}.")
    else:
        df = pd.read_csv(path)
        if {"x", "y"}.issubset(df.columns):
            arr = df[["x", "y"]].to_numpy(dtype=np.float32)
        elif {"pred_x", "pred_y"}.issubset(df.columns):
            arr = df[["pred_x", "pred_y"]].to_numpy(dtype=np.float32)
        else:
            raise ValueError(f"Prediction CSV must contain x,y or pred_x,pred_y columns: {path}")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Prediction points must have shape [T, >=2], got {arr.shape}.")
    arr = arr[:, :2]
    return arr[np.isfinite(arr).all(axis=1)]


def _plot_polyline(ax: plt.Axes, points: np.ndarray, color: str, label: str, *, linestyle: str = "-", zorder: int = 10) -> None:
    if len(points) == 0:
        return
    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linewidth=2.8,
        linestyle=linestyle,
        marker="o",
        markersize=2.6,
        markevery=max(len(points) // 8, 1),
        label=label,
        zorder=zorder,
        solid_capstyle="round",
    )
    ax.scatter(points[-1, 0], points[-1, 1], s=34, color=color, edgecolors="white", linewidths=0.7, zorder=zorder + 1)


def _draw_context(ax: plt.Axes, tracks: pd.DataFrame, frame_id: int, target_track_id: str) -> None:
    current = tracks[(tracks["frame_id"].astype(int) == frame_id) & (tracks["track_id"].map(str) != target_track_id)]
    if current.empty:
        return
    ax.scatter(current["x"], current["y"], s=14, color="#555555", alpha=0.55, linewidths=0, label="Context agents", zorder=7)


def _set_view(ax: plt.Axes, map_features: dict[str, dict[str, Any]], focus: list[np.ndarray]) -> None:
    focus = [points for points in focus if len(points)]
    if focus:
        points = np.concatenate(focus, axis=0)
    else:
        geometries = [_feature_geometry(feature)[0] for feature in map_features.values()]
        geometries = [points for points in geometries if len(points)]
        points = np.concatenate(geometries, axis=0) if geometries else np.zeros((1, 2), dtype=np.float32)
    x_min, y_min = np.nanmin(points, axis=0)
    x_max, y_max = np.nanmax(points, axis=0)
    padding = 14.0
    min_span = 45.0
    span = max(float(x_max - x_min), float(y_max - y_min), min_span)
    x_center = float((x_min + x_max) / 2.0)
    y_center = float((y_min + y_max) / 2.0)
    half = span / 2.0 + padding
    ax.set_xlim(x_center - half, x_center + half)
    ax.set_ylim(y_center - half, y_center + half)


def main() -> None:
    args = _parse_args()
    city = normalize_city_name(args.city)
    map_path = resolve_map_path(city, args.data_root, args.map_fallback_root)
    map_features, _ = parse_osm_map(map_path)

    tracks = None
    track_id = None
    frame_id = None
    past = np.zeros((0, 2), dtype=np.float32)
    gt = np.zeros((0, 2), dtype=np.float32)
    pred = np.zeros((0, 2), dtype=np.float32)

    if args.record or args.track_id:
        record = _find_record(args.data_root, args.map_fallback_root, city, args.record)
        if record is None:
            raise FileNotFoundError(f"No records found for city={city!r} under {args.data_root}")
        tracks = _combined_tracks(record)
        tracks["track_id"] = tracks["track_id"].map(str)
        track_id = str(args.track_id) if args.track_id is not None else _choose_track_id(tracks)
        track = tracks[tracks["track_id"] == track_id]
        if track.empty:
            raise ValueError(f"Track {track_id!r} was not found in record {record.record_name!r}.")
        frame_id = _choose_frame_id(track, args.past_len, args.future_len, args.frame_id)
        past, gt = _track_window_points(track, frame_id, args.past_len, args.future_len)

    if args.prediction_path:
        pred = _load_prediction(args.prediction_path, args.prediction_key)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_facecolor("#eeeeee")
    _draw_osm_map(ax, map_features)
    if tracks is not None and frame_id is not None and track_id is not None and args.show_context:
        _draw_context(ax, tracks, frame_id, track_id)
    _plot_polyline(ax, past, "#333333", "Past", linestyle="--", zorder=9)
    _plot_polyline(ax, gt, "#1f77b4", "Actual future", zorder=10)
    _plot_polyline(ax, pred, "#e4572e", "Predicted future", zorder=11)
    if len(past):
        ax.scatter(past[-1, 0], past[-1, 1], s=50, color="#111111", edgecolors="white", linewidths=0.8, label="Current", zorder=14)

    _set_view(ax, map_features, [past, gt, pred])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    title = args.title
    if title is None:
        parts = [city]
        if args.record:
            parts.append(args.record)
        if track_id is not None:
            parts.append(f"track {track_id}")
        if frame_id is not None:
            parts.append(f"frame {frame_id}")
        title = " | ".join(parts)
    ax.set_title(title, fontsize=12, pad=8)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right", frameon=True, framealpha=0.94, fontsize=9)
    fig.tight_layout(pad=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[done] saved visualization to {args.output}")


if __name__ == "__main__":
    main()
