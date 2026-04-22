import argparse
import math
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np
from PIL import Image


MAP_STYLE = {
    "LANE_SURFACE_STREET": {"color": "#d9d9d9", "linewidth": 1.2, "linestyle": "-"},
    "ROAD_LINE_BROKEN_SINGLE_WHITE": {"color": "#ffffff", "linewidth": 1.0, "linestyle": "--"},
    "ROAD_LINE_SOLID_SINGLE_WHITE": {"color": "#ffffff", "linewidth": 1.2, "linestyle": "-"},
    "ROAD_EDGE_BOUNDARY": {"color": "#6f6f6f", "linewidth": 1.4, "linestyle": "-"},
    "CROSSWALK": {"color": "#f3f3f3", "linewidth": 1.6, "linestyle": "-"},
}

TRACK_STYLE = {
    "VEHICLE": {"color": "#0077b6", "zorder": 5},
    "CYCLIST": {"color": "#2a9d8f", "zorder": 4},
    "PEDESTRIAN": {"color": "#e76f51", "zorder": 4},
}

LIGHT_STATE_COLOR = {
    "LANE_STATE_STOP": "#d62828",
    "LANE_STATE_CAUTION": "#f4a261",
    "LANE_STATE_GO": "#2a9d8f",
    "LANE_STATE_ARROW_STOP": "#d62828",
    "LANE_STATE_ARROW_CAUTION": "#f4a261",
    "LANE_STATE_ARROW_GO": "#2a9d8f",
    "LANE_STATE_FLASHING_STOP": "#d62828",
    "LANE_STATE_FLASHING_CAUTION": "#f4a261",
    "LANE_STATE_UNKNOWN": "#bdbdbd",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render an animated GIF from a ScenarioNet SinD scenario."
    )
    parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
        help="Path to a single ScenarioNet .pkl scenario file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output GIF path.",
    )
    parser.add_argument(
        "--agent-types",
        nargs="+",
        default=["VEHICLE"],
        help="Agent types to animate. Example: VEHICLE PEDESTRIAN CYCLIST",
    )
    parser.add_argument(
        "--tail-length",
        type=int,
        default=20,
        help="Number of historical frames to keep visible per agent.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the output GIF.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        default=[8.0, 8.0],
        metavar=("W", "H"),
        help="Figure size in inches.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=8.0,
        help="Extra margin around the scenario extent.",
    )
    parser.add_argument(
        "--show-traffic-lights",
        action="store_true",
        help="Render traffic light stop points using their per-frame state.",
    )
    return parser.parse_args()


def load_scenario(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def rotated_box(center_xy, heading, length, width):
    dx = length / 2.0
    dy = width / 2.0
    corners = np.array(
        [[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]],
        dtype=np.float32,
    )
    cos_h = math.cos(float(heading))
    sin_h = math.sin(float(heading))
    rotation = np.array([[cos_h, -sin_h], [sin_h, cos_h]], dtype=np.float32)
    return corners @ rotation.T + center_xy[None, :]


def iter_track_points(track):
    state = track["state"]
    positions = np.asarray(state["position"], dtype=np.float32)
    valid = np.asarray(state["valid"]).reshape(-1) > 0
    return positions[:, :2], valid


def compute_plot_bounds(scenario, selected_types, margin):
    xy_chunks = []

    for feature in scenario["map_features"].values():
        polyline = feature.get("polyline")
        polygon = feature.get("polygon")
        geom = polyline if polyline is not None else polygon
        if geom is not None and len(geom):
            xy_chunks.append(np.asarray(geom, dtype=np.float32)[:, :2])

    for track in scenario["tracks"].values():
        if track["type"] not in selected_types:
            continue
        positions, valid = iter_track_points(track)
        if valid.any():
            xy_chunks.append(positions[valid])

    if not xy_chunks:
        return (-10, 10), (-10, 10)

    all_xy = np.concatenate(xy_chunks, axis=0)
    min_xy = all_xy.min(axis=0) - margin
    max_xy = all_xy.max(axis=0) + margin
    return (float(min_xy[0]), float(max_xy[0])), (float(min_xy[1]), float(max_xy[1]))


def draw_map(ax, scenario):
    for feature in scenario["map_features"].values():
        geom = feature.get("polyline")
        if geom is None:
            geom = feature.get("polygon")
        if geom is None or len(geom) == 0:
            continue

        pts = np.asarray(geom, dtype=np.float32)[:, :2]
        style = MAP_STYLE.get(
            feature["type"],
            {"color": "#9a9a9a", "linewidth": 0.8, "linestyle": "-"},
        )
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style["linestyle"],
            alpha=0.9,
            zorder=1,
        )


def draw_traffic_lights(ax, scenario, frame_idx):
    for signal in scenario["dynamic_map_states"].values():
        stop_point = signal.get("stop_point")
        states = signal.get("state", {}).get("object_state", [])
        if stop_point is None or not states:
            continue
        state = states[min(frame_idx, len(states) - 1)]
        color = LIGHT_STATE_COLOR.get(state, "#bdbdbd")
        point = np.asarray(stop_point, dtype=np.float32)[:2]
        ax.scatter(
            point[0],
            point[1],
            s=36,
            color=color,
            edgecolors="black",
            linewidths=0.4,
            zorder=8,
        )


def draw_tracks(ax, scenario, frame_idx, selected_types, tail_length):
    tracks_to_predict = scenario["metadata"].get("tracks_to_predict", {})
    highlighted_ids = {str(v["track_id"]) for v in tracks_to_predict.values()}

    for track_id, track in scenario["tracks"].items():
        if track["type"] not in selected_types:
            continue

        positions = np.asarray(track["state"]["position"], dtype=np.float32)[:, :2]
        headings = np.asarray(track["state"]["heading"], dtype=np.float32)
        valid = np.asarray(track["state"]["valid"]).reshape(-1) > 0
        length = np.asarray(track["state"]["length"], dtype=np.float32).reshape(-1)
        width = np.asarray(track["state"]["width"], dtype=np.float32).reshape(-1)

        style = TRACK_STYLE.get(track["type"], {"color": "#4c78a8", "zorder": 5})
        color = "#ffb703" if str(track_id) in highlighted_ids else style["color"]
        zorder = 7 if str(track_id) in highlighted_ids else style["zorder"]

        start = max(0, frame_idx - tail_length + 1)
        tail_valid = valid[start : frame_idx + 1]
        tail_points = positions[start : frame_idx + 1]
        if tail_valid.any():
            tail_points = tail_points[tail_valid]
            if len(tail_points) >= 2:
                ax.plot(
                    tail_points[:, 0],
                    tail_points[:, 1],
                    color=color,
                    linewidth=2.0 if str(track_id) in highlighted_ids else 1.5,
                    alpha=0.85,
                    zorder=zorder,
                )
            ax.scatter(
                tail_points[:, 0],
                tail_points[:, 1],
                s=6,
                color=color,
                alpha=0.45,
                zorder=zorder,
            )

        if not valid[frame_idx]:
            continue

        center = positions[frame_idx]
        if track["type"] == "PEDESTRIAN":
            patch = Circle(
                center,
                radius=max(float(width[frame_idx]) / 2.0, 0.25),
                facecolor=color,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.95,
                zorder=zorder + 1,
            )
        else:
            corners = rotated_box(
                center_xy=center,
                heading=headings[frame_idx],
                length=max(float(length[frame_idx]), 0.8),
                width=max(float(width[frame_idx]), 0.4),
            )
            patch = Polygon(
                corners,
                closed=True,
                facecolor=color,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.95,
                zorder=zorder + 1,
            )
        ax.add_patch(patch)

        ax.text(
            center[0],
            center[1],
            str(track_id),
            fontsize=6,
            color="black",
            ha="center",
            va="center",
            zorder=zorder + 2,
        )


def figure_to_image(fig):
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return Image.fromarray(buffer[..., :3])


def render_gif(
    scenario,
    output_path: Path,
    selected_types,
    tail_length,
    fps,
    dpi,
    figsize,
    margin,
    show_traffic_lights,
):
    xlim, ylim = compute_plot_bounds(scenario, selected_types, margin)
    timestamps = scenario["metadata"]["ts"]
    scenario_id = scenario["metadata"].get("scenario_id", scenario["id"])

    frames = []
    for frame_idx, timestamp in enumerate(timestamps):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f7f7f7")

        draw_map(ax, scenario)
        if show_traffic_lights:
            draw_traffic_lights(ax, scenario, frame_idx)
        draw_tracks(ax, scenario, frame_idx, selected_types, tail_length)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_title(
            f"{scenario_id} | frame {frame_idx + 1}/{len(timestamps)} | t={float(timestamp):.2f}s",
            fontsize=11,
        )

        frames.append(figure_to_image(fig))
        plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(int(1000 / fps), 1)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def main():
    args = parse_args()
    scenario = load_scenario(args.scenario)
    render_gif(
        scenario=scenario,
        output_path=args.output,
        selected_types=set(args.agent_types),
        tail_length=args.tail_length,
        fps=args.fps,
        dpi=args.dpi,
        figsize=tuple(args.figsize),
        margin=args.margin,
        show_traffic_lights=args.show_traffic_lights,
    )
    print(f"Saved animation to {args.output}")


if __name__ == "__main__":
    main()
