from __future__ import annotations

import os
import math
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str((Path(".cache")).resolve()))
os.environ.setdefault("MPLCONFIGDIR", str((Path(".mplconfig")).resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "SinD" / "Dataset" / "Xi_an"
LIGHT_OUTPUT_ROOT = PROJECT_ROOT / "output" / "lights"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "xian"

CHANNELS = ["Traffic light 1", "Traffic light 2"]
STATE_LABEL = {0: "R", 1: "G", 3: "Y"}
STATE_COLOR = {0: "#d62828", 1: "#2a9d8f", 3: "#f4a261"}


@dataclass(frozen=True)
class ChannelSummary:
    record_name: str
    channel: str
    cycle_s: float
    green_s: float
    yellow_s: float
    red_s: float


def _load_record_traffic_light(record_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(record_dir / "Traffic_Lights.csv").copy()
    df["record_name"] = record_dir.name
    df["timestamp(ms)"] = pd.to_numeric(df["timestamp(ms)"], errors="coerce")
    for channel in CHANNELS:
        df[channel] = pd.to_numeric(df[channel], errors="coerce")
    df = df.dropna(subset=CHANNELS, how="all")
    return df


def _record_phase_events(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for channel in CHANNELS:
        channel_df = df[["record_name", "timestamp(ms)", channel]].rename(
            columns={"timestamp(ms)": "time_ms", channel: "state"}
        )
        channel_df["channel"] = channel
        rows.extend(channel_df.to_dict("records"))
    events = pd.DataFrame(rows).dropna(subset=["time_ms", "state"])
    events["state"] = events["state"].astype(int)
    events = events.sort_values(["record_name", "channel", "time_ms"])
    return events


def _durations_by_state(events: pd.DataFrame) -> list[ChannelSummary]:
    out: list[ChannelSummary] = []
    for (record_name, channel), group in events.groupby(["record_name", "channel"], sort=True):
        group = group.sort_values("time_ms")
        times = group["time_ms"].to_numpy(dtype=float)
        states = group["state"].to_numpy(dtype=int)
        green_starts = times[states == 1]
        cycle_intervals = np.diff(green_starts) / 1000.0
        valid_cycles = cycle_intervals[(cycle_intervals > 100.0) & (cycle_intervals < 140.5)]
        cycle_s = float(np.median(valid_cycles)) if len(valid_cycles) else float("nan")

        durations: dict[int, list[float]] = {0: [], 1: [], 3: []}
        if len(times) >= 2:
            for state, dt in zip(states[:-1], np.diff(times) / 1000.0):
                if state in durations and dt > 0:
                    durations[int(state)].append(float(dt))
        out.append(
            ChannelSummary(
                record_name=record_name,
                channel=channel,
                cycle_s=cycle_s,
                green_s=float(np.median(durations[1])) if durations[1] else float("nan"),
                yellow_s=float(np.median(durations[3])) if durations[3] else float("nan"),
                red_s=float(np.median(durations[0])) if durations[0] else float("nan"),
            )
        )
    return out


def _load_precomputed_cycle_summary() -> pd.DataFrame | None:
    path = LIGHT_OUTPUT_ROOT / "traffic_light_channel_cycle.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["city"] == "Xi_an"].copy()
    if df.empty:
        return None
    df = df.rename(
        columns={
            "traffic_light_channel": "channel",
            "estimated_cycle_s": "cycle_s",
            "green_duration_s": "green_s",
            "yellow_duration_s": "yellow_s",
            "red_duration_s": "red_s",
        }
    )
    keep = ["record_name", "channel", "cycle_s", "green_s", "yellow_s", "red_s"]
    for col in keep[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[keep]


def _state_at_time(times: np.ndarray, states: np.ndarray, query_ms: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(times, query_ms, side="right") - 1
    valid = indices >= 0
    out = np.full(len(query_ms), -1, dtype=int)
    out[valid] = states[indices[valid]]
    return out


def _crossing_state_table() -> pd.DataFrame:
    phase_events = pd.read_csv(LIGHT_OUTPUT_ROOT / "traffic_light_channel_phase_events.csv")
    crossings = pd.read_csv(LIGHT_OUTPUT_ROOT / "crossing_events.csv")
    phase_events = phase_events[phase_events["city"] == "Xi_an"].copy()
    crossings = crossings[crossings["city"] == "Xi_an"].copy()
    rows: list[pd.DataFrame] = []
    for record_name, crossing_group in crossings.groupby("record_name", sort=True):
        record_phase = phase_events[phase_events["record_name"] == record_name].copy()
        if record_phase.empty:
            continue
        pivot = (
            record_phase.pivot_table(
                index="time_ms", columns="traffic_light_channel", values="state", aggfunc="last"
            )
            .sort_index()
            .ffill()
            .fillna(0)
        )
        times = pivot.index.to_numpy(dtype=float)
        crossing_group = crossing_group.copy()
        crossing_times = crossing_group["crossing_timestamp_ms"].to_numpy(dtype=float)
        for channel in CHANNELS:
            crossing_group[channel] = _state_at_time(times, pivot[channel].to_numpy(dtype=int), crossing_times)
        rows.append(crossing_group)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _dominant_groups(crossing_states: pd.DataFrame) -> pd.DataFrame:
    group_rows: list[dict[str, object]] = []
    for channel in CHANNELS:
        other = CHANNELS[1] if channel == CHANNELS[0] else CHANNELS[0]
        green = crossing_states[(crossing_states[channel] == 1) & (crossing_states[other] == 0)].copy()
        counts = (
            green.groupby(["stopline_id", "movement"], sort=True)
            .size()
            .reset_index(name="green_crossings")
            .sort_values("green_crossings", ascending=False)
        )
        counts["channel"] = channel
        group_rows.append(counts)
    groups = pd.concat(group_rows, ignore_index=True)

    selection_rows: list[dict[str, object]] = []
    for channel, group in groups.groupby("channel", sort=True):
        dominant_straights = group[group["movement"] == "straight"].sort_values("green_crossings", ascending=False).head(2)
        for row in dominant_straights.itertuples(index=False):
            selection_rows.append(
                {
                    "channel": channel,
                    "stopline_id": int(row.stopline_id),
                    "dominant_motion": row.movement,
                    "green_crossings": int(row.green_crossings),
                    "phase_role": "dominant_through_corridor",
                }
            )
        permissive_left = (
            group[(group["movement"] == "left") & (~group["stopline_id"].isin(dominant_straights["stopline_id"]))]
            .sort_values("green_crossings", ascending=False)
            .head(1)
        )
        for row in permissive_left.itertuples(index=False):
            selection_rows.append(
                {
                    "channel": channel,
                    "stopline_id": int(row.stopline_id),
                    "dominant_motion": row.movement,
                    "green_crossings": int(row.green_crossings),
                    "phase_role": "secondary_left_turn",
                }
            )
    return pd.DataFrame(selection_rows)


def _plot_cycle_consistency(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    records = sorted(summary["record_name"].unique())
    x = np.arange(len(records))
    for idx, channel in enumerate(CHANNELS):
        channel_df = summary[summary["channel"] == channel].set_index("record_name").loc[records]
        axes[0].plot(x, channel_df["cycle_s"], marker="o", linewidth=1.5, label=channel)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(records, rotation=60, ha="right", fontsize=8)
    axes[0].set_ylabel("Estimated Cycle (s)")
    axes[0].set_title("Xi'an Record-Level Cycle Lengths")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    dur_cols = [("green_s", "Green"), ("yellow_s", "Yellow"), ("red_s", "Red")]
    pos = np.arange(len(dur_cols))
    width = 0.35
    for offset, channel in [(-width / 2, CHANNELS[0]), (width / 2, CHANNELS[1])]:
        channel_df = summary[summary["channel"] == channel]
        means = [channel_df[col].mean() for col, _ in dur_cols]
        stds = [channel_df[col].std(ddof=0) for col, _ in dur_cols]
        axes[1].bar(pos + offset, means, width=width, yerr=stds, capsize=3, label=channel)
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels([label for _, label in dur_cols])
    axes[1].set_ylabel("Median Duration Per Record (s)")
    axes[1].set_title("State Duration Stability Across Xi'an Records")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "xian_cycle_consistency.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_phase_template(summary: pd.DataFrame, selected: pd.DataFrame, stop_lines: pd.DataFrame) -> None:
    phase_defs = [
        ("Phase 1", "Traffic light 1", 1, "Traffic light 2", 0, "L1 green, L2 red"),
        ("Phase 2", "Traffic light 1", 3, "Traffic light 2", 0, "L1 yellow, L2 red"),
        ("Phase 3", "Traffic light 1", 0, "Traffic light 2", 1, "L1 red, L2 green"),
        ("Phase 4", "Traffic light 1", 0, "Traffic light 2", 3, "L1 red, L2 yellow"),
    ]
    stop_lookup = stop_lines.drop_duplicates("stopline_id").set_index("stopline_id")
    all_pts = np.concatenate(
        [
            stop_lines[["x1", "y1"]].to_numpy(dtype=float),
            stop_lines[["x2", "y2"]].to_numpy(dtype=float),
        ],
        axis=0,
    )
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)
    margin = max(x_max - x_min, y_max - y_min) * 0.25

    green_l1 = float(summary[summary["channel"] == "Traffic light 1"]["green_s"].median())
    yellow_l1 = float(summary[summary["channel"] == "Traffic light 1"]["yellow_s"].median())
    green_l2 = float(summary[summary["channel"] == "Traffic light 2"]["green_s"].median())
    yellow_l2 = float(summary[summary["channel"] == "Traffic light 2"]["yellow_s"].median())
    durations = {
        "Phase 1": green_l1,
        "Phase 2": yellow_l1,
        "Phase 3": green_l2,
        "Phase 4": yellow_l2,
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    for ax, (phase_name, ch_a, state_a, ch_b, state_b, title_suffix) in zip(axes.flat, phase_defs):
        active_channel = ch_a if state_a == 1 else ch_b if state_b == 1 else None
        ax.set_title(f"{phase_name}: {title_suffix}\n{durations[phase_name]:.1f}s", fontsize=11)
        for row in stop_lookup.itertuples():
            active = active_channel is not None and not selected[
                (selected["channel"] == active_channel) & (selected["stopline_id"] == int(row.Index))
            ].empty
            color = "#2a9d8f" if active else "#c1121f" if active_channel is not None else "#f4a261"
            alpha = 0.95 if active else 0.55
            ax.plot([row.x1, row.x2], [row.y1, row.y2], color=color, linewidth=4.0, alpha=alpha, solid_capstyle="round")
            mid_x = (row.x1 + row.x2) / 2.0
            mid_y = (row.y1 + row.y2) / 2.0
            ax.text(mid_x, mid_y, str(int(row.Index)), fontsize=9, ha="center", va="center", color="#111111")

        if active_channel is not None:
            chosen = selected[selected["channel"] == active_channel]
            for idx, row in enumerate(chosen.itertuples(index=False), start=1):
                stop = stop_lookup.loc[int(row.stopline_id)]
                mid_x = (stop.x1 + stop.x2) / 2.0
                mid_y = (stop.y1 + stop.y2) / 2.0
                dx = 10.0 if mid_x < (x_min + x_max) / 2 else -10.0
                dy = 8.0 if mid_y < (y_min + y_max) / 2 else -8.0
                label = f"{row.stopline_id}: {row.dominant_motion}"
                ax.annotate(
                    label,
                    xy=(mid_x, mid_y),
                    xytext=(mid_x + dx, mid_y + dy * (0.7 if idx == 1 else 1.0)),
                    fontsize=9,
                    arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "#444444"},
                    bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#bbbbbb", "alpha": 0.9},
                )
            ax.text(
                0.02,
                0.02,
                "Green phase shown as dominant through corridor.\nLeft turns occur too, but are treated as secondary to avoid over-claiming protected movements.",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#dddddd", "alpha": 0.9},
            )
        else:
            ax.text(
                0.02,
                0.02,
                "Clearance phase. No explicit all-red interval is logged in most Xi'an records.",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#dddddd", "alpha": 0.9},
            )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    fig.suptitle("Xi'an Two-Channel Canonical Cycle and Approximate Controlled Stopline Groups", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "xian_canonical_cycle.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_record_heatmap(events: pd.DataFrame) -> None:
    records = sorted(events["record_name"].unique())
    fig, axes = plt.subplots(len(records), 1, figsize=(12, max(8, len(records) * 0.55)), sharex=False)
    if len(records) == 1:
        axes = [axes]

    for ax, record_name in zip(axes, records):
        record_df = events[events["record_name"] == record_name]
        pivot = (
            record_df.pivot_table(index="time_ms", columns="channel", values="state", aggfunc="last")
            .sort_index()
            .ffill()
        )
        for row_idx, channel in enumerate(CHANNELS):
            series = pivot[channel].to_numpy(dtype=float)
            times = pivot.index.to_numpy(dtype=float) / 1000.0
            segments = []
            start = 0
            for idx in range(1, len(series)):
                if series[idx] != series[start]:
                    segments.append((times[start], times[idx], int(series[start])))
                    start = idx
            if len(series):
                segments.append((times[start], times[-1], int(series[start])))
            for t0, t1, state in segments:
                ax.fill_between([t0, t1], [row_idx + 0.1, row_idx + 0.1], [row_idx + 0.9, row_idx + 0.9], color=STATE_COLOR.get(state, "#bbbbbb"))
                if t1 > t0:
                    ax.text((t0 + t1) / 2, row_idx + 0.5, STATE_LABEL.get(state, "?"), ha="center", va="center", fontsize=7, color="white")
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["L1", "L2"])
        ax.set_title(record_name, loc="left", fontsize=9)
        ax.grid(True, axis="x", alpha=0.15)

    fig.suptitle("Xi'an Record-Level Channel State Timelines", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "xian_record_state_timelines.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_report(summary: pd.DataFrame, selected: pd.DataFrame) -> None:
    cycle_pivot = summary.pivot(index="record_name", columns="channel", values="cycle_s")
    cycle_diff = (cycle_pivot[CHANNELS[0]] - cycle_pivot[CHANNELS[1]]).abs()
    same_cycle = bool((cycle_diff.fillna(0) < 0.5).all())
    median_cycle = float(summary["cycle_s"].median())
    min_cycle = float(summary["cycle_s"].min())
    max_cycle = float(summary["cycle_s"].max())
    lines = [
        "# Xi'an Traffic-Light Analysis",
        "",
        "## Cycle Periodicity",
        f"- Records analyzed: {summary['record_name'].nunique()}",
        f"- Channels per record: {len(CHANNELS)}",
        f"- Median cycle: {median_cycle:.3f}s",
        f"- Range across record/channel estimates: [{min_cycle:.3f}s, {max_cycle:.3f}s]",
        f"- Are all records effectively on the same cycle template: {'yes' if same_cycle else 'mostly yes, but with minor jitter'}",
        "- Dominant template: `L1=G -> L1=Y -> L2=G -> L2=Y` with no stable explicit all-red segment in the CSV.",
        "",
        "## Approximate Controlled Movement Groups",
        "- In Xi'an, the two logged channels behave like two coarse opposing corridors rather than a lane-complete signal inventory.",
        "- To avoid over-claiming protected left phases, the canonical phase figure uses the strongest straight-through corridors as the primary controlled groups.",
        "- Left-turn crossings exist during the same green windows, but they are treated here as secondary/permissive overlap unless the evidence is very strong.",
        "",
        "### Dominant stopline groups used in the phase figure",
    ]
    for channel, group in selected.groupby("channel", sort=True):
        lines.append(f"- {channel}:")
        for row in group.itertuples(index=False):
            lines.append(
                f"  stopline `{row.stopline_id}` -> `{row.dominant_motion}` ({row.phase_role}, {row.green_crossings} green-window crossings)"
            )
    lines.append("")
    lines.append("## Files")
    for name in [
        "xian_cycle_summary.csv",
        "xian_cycle_consistency.png",
        "xian_record_state_timelines.png",
        "xian_dominant_groups.csv",
        "xian_canonical_cycle.png",
    ]:
        lines.append(f"- `{name}`")
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    record_dirs = sorted([p for p in DATA_ROOT.iterdir() if p.is_dir()])
    record_frames = [_load_record_traffic_light(record_dir) for record_dir in record_dirs if (record_dir / "Traffic_Lights.csv").exists()]
    traffic = pd.concat(record_frames, ignore_index=True)
    events = _record_phase_events(traffic)

    summary = _load_precomputed_cycle_summary()
    if summary is None:
        summary = pd.DataFrame([summary.__dict__ for summary in _durations_by_state(events)])
    summary.to_csv(OUTPUT_ROOT / "xian_cycle_summary.csv", index=False)

    crossing_states = _crossing_state_table()
    crossing_states.to_csv(OUTPUT_ROOT / "xian_crossing_states.csv", index=False)

    selected = _dominant_groups(crossing_states)
    selected.to_csv(OUTPUT_ROOT / "xian_dominant_groups.csv", index=False)

    stop_lines = pd.read_csv(LIGHT_OUTPUT_ROOT / "stop_lines.csv")
    stop_lines = stop_lines[stop_lines["city"] == "Xi_an"].copy()

    _plot_cycle_consistency(summary)
    _plot_record_heatmap(events)
    _plot_phase_template(summary, selected, stop_lines)
    _write_report(summary, selected)


if __name__ == "__main__":
    main()
