"""Visualize config signal bindings as a single static intersection map per city.

Draws the full road network (lanes + connectors + stop lines) from OSM, then
color-codes every bound lane and stopline by its signal channel group. This
shows at a glance which lanes share a signal and whether the spatial grouping
makes sense.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCENARIONET_ROOT = PROJECT_ROOT / "scenarionet"
for path in (PROJECT_ROOT, SCENARIONET_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sind_converter.data.discovery import discover_records, normalize_city_name
from sind_converter.lights.stopline_extraction import extract_stop_lines
from sind_converter.maps.osm import parse_osm_map

# ── colour constants ────────────────────────────────────────────────────────
BASE_LANE_COLOR = "#d9ded9"
BASE_CONNECTOR_COLOR = "#b8c4b8"
BASE_STOPLINE_COLOR = "#62686f"
NO_SIGNAL_COLOR = "#d62828"  # red for unbound stoplines

ALL_CITIES = ["Changchun", "Chongqing", "Tianjin", "Xi_an"]


# ── helpers ──────────────────────────────────────────────────────────────────
def _channel_short_name(name: str) -> str:
    match = re.search(r"(\d+)", str(name))
    if match:
        return f"L{match.group(1)}"
    return str(name)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_lane_feature_id(value: str) -> str:
    return value if value.startswith("lane_") else f"lane_{value}"


def _arrow_at_midpoint(ax: plt.Axes, polyline: np.ndarray, color: str) -> None:
    """Draw a small direction-of-travel arrow at the midpoint of *polyline*."""
    if polyline.shape[0] < 2:
        return
    mid = len(polyline) // 2
    p_before = polyline[max(mid - 1, 0), :2]
    p_after = polyline[min(mid + 1, len(polyline) - 1), :2]
    direction = p_after - p_before
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return
    direction = direction / norm
    pos = polyline[mid, :2]
    arrow_len = 1.8
    ax.annotate(
        "",
        xy=(pos[0] + direction[0] * arrow_len / 2, pos[1] + direction[1] * arrow_len / 2),
        xytext=(pos[0] - direction[0] * arrow_len / 2, pos[1] - direction[1] * arrow_len / 2),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, mutation_scale=14),
        zorder=9,
    )


# ── base map drawing (reused from visualize_sind_signal_bindings.py) ─────────
def _draw_base_map(
    ax: plt.Axes,
    map_features: dict[str, dict[str, Any]],
    stop_lines_df: pd.DataFrame,
) -> None:
    for feature_id, feature in map_features.items():
        if not str(feature_id).startswith("lane_"):
            continue
        polyline = np.asarray(feature.get("polyline", np.zeros((0, 3), dtype=np.float32)), dtype=float)
        if polyline.ndim != 2 or polyline.shape[0] == 0:
            continue
        color = BASE_CONNECTOR_COLOR if "_to_" in str(feature_id) else BASE_LANE_COLOR
        linewidth = 2.8 if "_to_" in str(feature_id) else 1.8
        ax.plot(
            polyline[:, 0],
            polyline[:, 1],
            color=color,
            linewidth=linewidth,
            alpha=0.8,
            solid_capstyle="round",
            zorder=1,
        )
    for line in stop_lines_df.itertuples(index=False):
        ax.plot(
            [line.x1, line.x2],
            [line.y1, line.y2],
            color=BASE_STOPLINE_COLOR,
            linewidth=3.0,
            alpha=0.85,
            zorder=3,
        )
        ax.text(
            float(line.mid_x),
            float(line.mid_y),
            str(line.stopline_id),
            fontsize=7,
            ha="center",
            va="center",
            color="#40464d",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.65, pad=0.8),
            zorder=4,
        )


# ── main plotting routine ───────────────────────────────────────────────────
def _plot_city_config_bindings(
    city: str,
    map_features: dict[str, dict[str, Any]],
    stop_lines_df: pd.DataFrame,
    channel_groups: list[dict[str, Any]],
    lane_bindings: list[dict[str, Any]],
    output_path: Path,
) -> None:
    cmap = plt.colormaps["tab10"]

    # Build lookup: (stopline_id, movement) -> lane_binding row
    lane_binding_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in lane_bindings:
        key = (str(row["stopline_id"]), str(row["movement"]).strip().lower())
        lane_binding_lookup[key] = row

    # Assign group index -> colour
    group_colors: dict[int, tuple[float, ...]] = {}
    for idx in range(len(channel_groups)):
        group_colors[idx] = cmap(idx % 10)

    # ── figure setup ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 14))

    # Compute plot limits from all points
    all_points: list[np.ndarray] = []
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
    margin = max(x_max - x_min, y_max - y_min, 1.0) * 0.08

    # 1. Base layer
    _draw_base_map(ax, map_features, stop_lines_df)

    # Track which (stopline_id, movement) are covered by channel_groups
    bound_keys: set[tuple[str, str]] = set()
    # Track which lane feature IDs have been drawn in colour
    drawn_lanes: set[str] = set()
    # Legend entries
    legend_handles: list[mpatches.Patch] = []
    # Movements in channel_groups with no matching lane_binding
    unmatched_group_movements: list[str] = []

    # 2 & 3. Bound lanes + stoplines overlay
    for group_idx, group in enumerate(channel_groups):
        color_rgb = group_colors[group_idx]
        color_hex = "#{:02x}{:02x}{:02x}".format(
            int(color_rgb[0] * 255), int(color_rgb[1] * 255), int(color_rgb[2] * 255)
        )
        channel_labels = [_channel_short_name(ch) for ch in group["traffic_light_channels"]]
        legend_label = f"{group['group_id']}  ({', '.join(channel_labels)})"
        legend_handles.append(mpatches.Patch(color=color_hex, label=legend_label))

        for movement_row in group["movements"]:
            stopline_id = str(movement_row["stopline_id"])
            movement = str(movement_row["movement"]).strip().lower()
            key = (stopline_id, movement)
            bound_keys.add(key)

            lane_row = lane_binding_lookup.get(key)
            if lane_row is None:
                unmatched_group_movements.append(f"{stopline_id}/{movement}")
                continue

            # Draw lanes
            for lane_id in lane_row["lane_ids"]:
                norm_id = _normalize_lane_feature_id(lane_id)
                feature = map_features.get(norm_id)
                if feature is None:
                    continue
                polyline = np.asarray(
                    feature.get("polyline", np.zeros((0, 3), dtype=np.float32)),
                    dtype=float,
                )
                if polyline.ndim != 2 or polyline.shape[0] == 0:
                    continue
                ax.plot(
                    polyline[:, 0],
                    polyline[:, 1],
                    color=color_hex,
                    linewidth=4.5,
                    alpha=0.9,
                    solid_capstyle="round",
                    zorder=5,
                )
                drawn_lanes.add(norm_id)
                _arrow_at_midpoint(ax, polyline, color_hex)

            # Draw stopline overlay
            stopline_df = stop_lines_df[stop_lines_df["stopline_id"].astype(str) == stopline_id]
            if not stopline_df.empty:
                row = stopline_df.iloc[0]
                ax.plot(
                    [row["x1"], row["x2"]],
                    [row["y1"], row["y2"]],
                    color=color_hex,
                    linewidth=5.5,
                    alpha=0.95,
                    solid_capstyle="round",
                    zorder=6,
                )
                # 4. Channel label at stopline midpoint
                short_ch = channel_labels[0]
                label_text = f"{short_ch}\u2192{movement}"
                ax.text(
                    float(row["mid_x"]),
                    float(row["mid_y"]) + 2.0,
                    label_text,
                    fontsize=9,
                    ha="center",
                    va="bottom",
                    color=color_hex,
                    fontweight="bold",
                    bbox=dict(
                        facecolor="white",
                        edgecolor=color_hex,
                        alpha=0.85,
                        pad=1.2,
                        linewidth=1.4,
                    ),
                    zorder=8,
                )

    # 6. Unbound stoplines / lanes — those in lane_bindings but NOT in channel_groups
    for key, lane_row in lane_binding_lookup.items():
        if key in bound_keys:
            continue
        stopline_id, movement = key
        # Draw lanes with dashed red stopline style
        for lane_id in lane_row["lane_ids"]:
            norm_id = _normalize_lane_feature_id(lane_id)
            feature = map_features.get(norm_id)
            if feature is None:
                continue
            polyline = np.asarray(
                feature.get("polyline", np.zeros((0, 3), dtype=np.float32)),
                dtype=float,
            )
            if polyline.ndim != 2 or polyline.shape[0] == 0:
                continue
            ax.plot(
                polyline[:, 0],
                polyline[:, 1],
                color=NO_SIGNAL_COLOR,
                linewidth=3.0,
                alpha=0.5,
                linestyle="--",
                solid_capstyle="round",
                zorder=5,
            )

        # Dashed red stopline
        stopline_df = stop_lines_df[stop_lines_df["stopline_id"].astype(str) == stopline_id]
        if not stopline_df.empty:
            row = stopline_df.iloc[0]
            ax.plot(
                [row["x1"], row["x2"]],
                [row["y1"], row["y2"]],
                color=NO_SIGNAL_COLOR,
                linewidth=4.5,
                alpha=0.85,
                linestyle="--",
                solid_capstyle="round",
                zorder=6,
            )
            ax.text(
                float(row["mid_x"]),
                float(row["mid_y"]) + 2.0,
                f"\u26a0 no signal\n{stopline_id}/{movement}",
                fontsize=8,
                ha="center",
                va="bottom",
                color=NO_SIGNAL_COLOR,
                fontweight="bold",
                bbox=dict(
                    facecolor="white",
                    edgecolor=NO_SIGNAL_COLOR,
                    alpha=0.85,
                    pad=1.0,
                    linewidth=1.2,
                ),
                zorder=8,
            )

    # Add legend entries for issues
    if unmatched_group_movements:
        issue_text = "No lane binding: " + ", ".join(unmatched_group_movements)
        legend_handles.append(
            mpatches.Patch(
                facecolor="none",
                edgecolor="orange",
                linewidth=1.5,
                linestyle="--",
                label=issue_text,
            )
        )

    # Unbound indicator in legend
    unbound_count = sum(1 for key in lane_binding_lookup if key not in bound_keys)
    if unbound_count > 0:
        legend_handles.append(
            mpatches.Patch(
                facecolor="none",
                edgecolor=NO_SIGNAL_COLOR,
                linewidth=1.5,
                linestyle="--",
                label=f"No signal group ({unbound_count} movement(s))",
            )
        )

    # 5. Legend
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    # Axes
    ax.set_aspect("equal")
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(False)

    # 7. Title
    ax.set_title(f"{city} \u2014 Config Signal Bindings", fontsize=14, fontweight="bold")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize config signal bindings as a single static intersection map per city.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "SinD" / "Dataset",
    )
    parser.add_argument(
        "--map-fallback-root",
        type=Path,
        default=PROJECT_ROOT / "SinD" / "Data",
    )
    parser.add_argument(
        "--binding-root",
        type=Path,
        default=PROJECT_ROOT / "sind_converter" / "lights" / "config",
    )
    parser.add_argument(
        "--figure-root",
        type=Path,
        default=PROJECT_ROOT / "output" / "config_binding_maps",
    )
    parser.add_argument(
        "--cities",
        nargs="*",
        default=ALL_CITIES,
    )
    args = parser.parse_args()

    # Discover all records (don't pass cities= to avoid directory-name mismatch
    # for Xi'an vs Xi_an), then filter by normalized city name.
    records = discover_records(args.data_root, args.map_fallback_root)
    requested = set(normalize_city_name(c) for c in args.cities)
    record_by_city: dict[str, Any] = {}
    for record in records:
        if requested and normalize_city_name(record.city) not in requested:
            continue
        record_by_city.setdefault(record.city, record)

    for city in args.cities:
        record = record_by_city.get(city)
        if record is None:
            print(f"[skip] no record discovered for city={city}")
            continue

        cg_path = args.binding_root / "channel_groups" / f"{city}.json"
        lb_path = args.binding_root / "lane_bindings" / f"{city}.json"
        if not cg_path.exists() or not lb_path.exists():
            print(f"[skip] missing config for city={city}: cg={cg_path.exists()} lb={lb_path.exists()}")
            continue

        channel_groups = _load_json(cg_path).get("channel_groups", [])
        lane_bindings = _load_json(lb_path).get("lane_bindings", [])
        if not channel_groups or not lane_bindings:
            print(f"[skip] empty bindings for city={city}")
            continue

        map_features, _ = parse_osm_map(record.map_path)
        stop_lines = extract_stop_lines(record.map_path, city)
        stop_lines_df = pd.DataFrame([line.as_row() for line in stop_lines])

        output_path = args.figure_root / f"{city.lower()}_binding_map.png"
        _plot_city_config_bindings(
            city,
            map_features,
            stop_lines_df,
            channel_groups,
            lane_bindings,
            output_path,
        )
        print(f"[ok] {city} -> {output_path}")


if __name__ == "__main__":
    main()
