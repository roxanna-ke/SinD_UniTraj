from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "SinD" / "Dataset"
OUTPUT_ROOT = PROJECT_ROOT / "output" / "light_cycle_analysis"

STATE_COLOR = {0: "#d62828", 1: "#2a9d8f", 3: "#f4a261", -1: "#bbbbbb"}
STATE_LABEL = {0: "R", 1: "G", 3: "Y", -1: "U"}


@dataclass(frozen=True)
class RecordSummary:
    city: str
    record_name: str
    channel: str
    cycle_s: float
    green_s: float
    yellow_s: float
    red_s: float


def _find_traffic_light_csv(record_dir: Path) -> Path | None:
    candidates = sorted(
        [
            *record_dir.glob("Traffic*.csv"),
            *record_dir.glob("*traffic*.csv"),
            *record_dir.glob("*Light*.csv"),
        ]
    )
    for path in candidates:
        if path.name not in {"Veh_smoothed_tracks.csv", "Ped_smoothed_tracks.csv"}:
            return path
    return None


def _timestamp_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        normalized = col.lower().replace(" ", "").replace("_", "")
        if normalized in {"timestamp(ms)", "timestampms"}:
            return col
    return None


def _raw_frame_time_axis(df: pd.DataFrame) -> np.ndarray:
    raw_frame_col = "RawFrameID" if "RawFrameID" in df.columns else None
    ts_col = _timestamp_column(df)
    if raw_frame_col is None and ts_col is None:
        return np.asarray([], dtype=float)
    if raw_frame_col is None:
        return pd.to_numeric(df[ts_col], errors="coerce").to_numpy(dtype=float)

    raw_frame = pd.to_numeric(df[raw_frame_col], errors="coerce").to_numpy(dtype=float)
    raw_time = raw_frame * (100.0 / 3.0)
    if ts_col is None:
        return raw_time - np.nanmin(raw_time)

    timestamp = pd.to_numeric(df[ts_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(raw_time) & np.isfinite(timestamp) & (timestamp > 0)
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(raw_frame[valid], timestamp[valid], 1)
        if np.isfinite(slope) and np.isfinite(intercept) and slope > 0:
            return raw_frame * float(slope) + float(intercept)
    return raw_time - np.nanmin(raw_time)


def _load_record(record_dir: Path) -> pd.DataFrame | None:
    csv_path = _find_traffic_light_csv(record_dir)
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path).copy()
    if df.empty:
        return None
    df["record_name"] = record_dir.name
    df["csv_name"] = csv_path.name
    return df


def _extract_light_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if "traffic light" in col.lower() and col not in {"RawFrameID", "timestamp(ms)"}
    ]


def _channel_events(df: pd.DataFrame, channel: str) -> pd.DataFrame:
    times = _raw_frame_time_axis(df)
    rows = pd.DataFrame(
        {
            "time_ms": times,
            "state": pd.to_numeric(df[channel], errors="coerce").fillna(-1).astype(int),
        }
    )
    rows = rows[np.isfinite(rows["time_ms"])].copy()
    rows["channel"] = channel
    return rows.sort_values("time_ms")


def _channel_summary(df: pd.DataFrame, city: str, record_name: str, channel: str) -> RecordSummary:
    events = _channel_events(df, channel)
    times = events["time_ms"].to_numpy(dtype=float)
    states = events["state"].to_numpy(dtype=int)

    durations: dict[int, list[float]] = {0: [], 1: [], 3: []}
    if len(times) >= 2:
        for state, dt in zip(states[:-1], np.diff(times) / 1000.0):
            if state in durations and dt > 0:
                durations[int(state)].append(float(dt))

    green_starts = times[states == 1] / 1000.0
    cycle_intervals = np.diff(green_starts)
    valid_cycles = cycle_intervals[(cycle_intervals > 30.0) & (cycle_intervals < 300.0)]
    cycle_s = float(np.median(valid_cycles)) if len(valid_cycles) else float("nan")

    return RecordSummary(
        city=city,
        record_name=record_name,
        channel=channel,
        cycle_s=cycle_s,
        green_s=float(np.median(durations[1])) if durations[1] else float("nan"),
        yellow_s=float(np.median(durations[3])) if durations[3] else float("nan"),
        red_s=float(np.median(durations[0])) if durations[0] else float("nan"),
    )


def _plot_city(summary: pd.DataFrame, city: str, output_dir: Path) -> None:
    records = sorted(summary["record_name"].unique())
    channels = sorted(summary["channel"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.6, 1.0]})

    x = np.arange(len(records))
    for channel in channels:
        channel_df = summary[summary["channel"] == channel].set_index("record_name").reindex(records)
        axes[0].plot(x, channel_df["cycle_s"], marker="o", linewidth=1.5, label=channel)
    axes[0].set_title(f"{city} Record-Level Cycle Lengths")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(records, rotation=60, ha="right", fontsize=8)
    axes[0].set_ylabel("Estimated Cycle (s)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, ncols=1)

    dur_cols = [("green_s", "Green"), ("yellow_s", "Yellow"), ("red_s", "Red")]
    pos = np.arange(len(dur_cols))
    width = min(0.8 / max(len(channels), 1), 0.28)
    offsets = np.linspace(-width * (len(channels) - 1) / 2, width * (len(channels) - 1) / 2, len(channels))
    for offset, channel in zip(offsets, channels):
        channel_df = summary[summary["channel"] == channel]
        means = [channel_df[col].mean() for col, _ in dur_cols]
        stds = [channel_df[col].std(ddof=0) for col, _ in dur_cols]
        axes[1].bar(pos + offset, means, width=width, yerr=stds, capsize=3, label=channel)
    axes[1].set_xticks(pos)
    axes[1].set_xticklabels([label for _, label in dur_cols])
    axes[1].set_title(f"{city} State Duration Stability")
    axes[1].set_ylabel("Median Duration Per Record (s)")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / f"{city.lower()}_cycle_consistency.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_sample_timeline(sample_df: pd.DataFrame, city: str, output_dir: Path) -> None:
    if sample_df.empty:
        return
    channels = sorted(_extract_light_columns(sample_df))
    if not channels:
        return

    fig, ax = plt.subplots(figsize=(14, max(4, 0.42 * len(channels))))
    y_positions = np.arange(len(channels))
    for y, channel in zip(y_positions, channels):
        events = _channel_events(sample_df, channel)
        times = events["time_ms"].to_numpy(dtype=float) / 1000.0
        states = events["state"].to_numpy(dtype=int)
        if len(times) == 0:
            continue
        start_idx = 0
        for idx in range(1, len(times)):
            if states[idx] != states[start_idx]:
                ax.fill_between(
                    [times[start_idx], times[idx]],
                    [y - 0.35, y - 0.35],
                    [y + 0.35, y + 0.35],
                    color=STATE_COLOR.get(int(states[start_idx]), "#bbbbbb"),
                )
                ax.text(
                    (times[start_idx] + times[idx]) / 2.0,
                    y,
                    STATE_LABEL.get(int(states[start_idx]), "?"),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                )
                start_idx = idx
        ax.fill_between(
            [times[start_idx], times[-1]],
            [y - 0.35, y - 0.35],
            [y + 0.35, y + 0.35],
            color=STATE_COLOR.get(int(states[start_idx]), "#bbbbbb"),
        )
        ax.text(
            (times[start_idx] + times[-1]) / 2.0,
            y,
            STATE_LABEL.get(int(states[start_idx]), "?"),
            ha="center",
            va="center",
            fontsize=7,
            color="white",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(channels, fontsize=7)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{city} Representative Channel Timelines: {sample_df['record_name'].iloc[0]}")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / f"{city.lower()}_sample_timeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze_city(city: str) -> pd.DataFrame:
    city_root = DATA_ROOT / city
    if not city_root.exists():
        raise FileNotFoundError(f"Missing city root: {city_root}")

    records = sorted([p for p in city_root.iterdir() if p.is_dir()])
    summaries: list[RecordSummary] = []
    sample_df: pd.DataFrame | None = None
    sample_record: str | None = None

    for record_dir in records:
        df = _load_record(record_dir)
        if df is None:
            continue
        channels = _extract_light_columns(df)
        if not channels:
            continue
        if sample_df is None:
            sample_df = df
            sample_record = record_dir.name
        for channel in channels:
            summaries.append(_channel_summary(df, city, record_dir.name, channel))

    if not summaries:
        raise ValueError(f"No traffic-light CSVs found for city={city}")

    output_dir = OUTPUT_ROOT / city
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame([s.__dict__ for s in summaries]).sort_values(["record_name", "channel"])
    summary_df.to_csv(output_dir / f"{city.lower()}_cycle_summary.csv", index=False)

    _plot_city(summary_df, city, output_dir)
    if sample_df is not None:
        _plot_sample_timeline(sample_df, city, output_dir)

    # city-level note
    cycle_pivot = summary_df.pivot(index="record_name", columns="channel", values="cycle_s")
    cycle_spread = cycle_pivot.max(axis=1) - cycle_pivot.min(axis=1)
    report = [
        f"# {city} Traffic-Light Periodicity",
        "",
        f"- Records analyzed: {summary_df['record_name'].nunique()}",
        f"- Channels analyzed: {summary_df['channel'].nunique()}",
        f"- Median cycle: {summary_df['cycle_s'].median():.3f}s",
        f"- Cycle range across records: [{summary_df['cycle_s'].min():.3f}s, {summary_df['cycle_s'].max():.3f}s]",
        f"- Median within-record channel spread: {cycle_spread.median():.3f}s",
        "",
        "## Files",
        f"- `{city.lower()}_cycle_summary.csv`",
        f"- `{city.lower()}_cycle_consistency.png`",
        f"- `{city.lower()}_sample_timeline.png`",
    ]
    (output_dir / "README.md").write_text("\n".join(report), encoding="utf-8")
    return summary_df


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cities = ["Tianjin", "Changchun", "Chongqing"]
    all_rows = []
    for city in cities:
        summary = analyze_city(city)
        all_rows.append(summary)
        print(f"[done] {city}: {len(summary)} channel summaries")

    pd.concat(all_rows, ignore_index=True).to_csv(OUTPUT_ROOT / "all_city_cycle_summary.csv", index=False)


if __name__ == "__main__":
    main()
