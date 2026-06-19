"""Compare inferred signal states against ground-truth traffic-light CSV data.

Reads the 7 CSV files produced by ``infer_sind_traffic_light_mapping.py``
(stored in ``output/lights/``) plus raw traffic-light CSVs via
``load_traffic_light_channels()``.  Produces per-pair timeline comparisons,
city overview grids, cycle comparison charts, an offset distribution histogram,
and quantitative metrics CSVs.
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))
os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sind_converter.data.discovery import RecordDescription, discover_records
from sind_converter.lights.channel_matching import (
    STATE_GREEN,
    STATE_RED,
    STATE_UNKNOWN,
    STATE_YELLOW,
    load_traffic_light_channels,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATE_COLOR = {0: "#d62828", 1: "#2a9d8f", 3: "#f4a261", -1: "#bbbbbb"}
STATE_LABEL = {0: "R", 1: "G", 3: "Y", -1: "U"}

# Binary comparison colours
COLOR_AGREE = "#2a9d8f"
COLOR_FN = "#e76f51"      # GT=G, inferred=red  (false negative)
COLOR_FP = "#6a4c93"      # GT≠G, inferred=green (false positive)
COLOR_UNKNOWN = "#bbbbbb"

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _safe_float(val: Any, default: float = float("nan")) -> float:
    if val is None or val == "" or (isinstance(val, float) and math.isnan(val)):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Record selection
# ---------------------------------------------------------------------------


def _pick_best_record(mapping: pd.DataFrame, city: str) -> str | None:
    """Pick the record with the most high/medium-confidence matched pairs."""
    city_map = mapping[
        (mapping["city"] == city) & (mapping["confidence"].isin(["high", "medium"]))
    ]
    if city_map.empty:
        return None
    counts = city_map.groupby("record_name").size()
    return counts.idxmax()


def _records_for_city(
    records: list[RecordDescription],
    city: str,
    record_names: list[str] | None,
    all_records: bool,
    mapping: pd.DataFrame,
) -> list[RecordDescription]:
    city_records = [r for r in records if r.city == city]
    if record_names:
        return [r for r in city_records if r.record_name in record_names]
    if all_records:
        return city_records
    best = _pick_best_record(mapping, city)
    if best is None:
        return city_records[:1] if city_records else []
    return [r for r in city_records if r.record_name == best]


# ---------------------------------------------------------------------------
# GT binary state series
# ---------------------------------------------------------------------------


def _gt_binary_at_seconds(
    channel_events: pd.DataFrame,
    t_start_s: float,
    t_end_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_seconds, binary_state) at 1-second resolution.

    binary_state: 1 = green, 0 = non-green (red/yellow), -1 = unknown
    (before first GT event).
    """
    if channel_events.empty:
        times_s = np.arange(t_start_s, t_end_s, 1.0)
        return times_s, np.full(len(times_s), -1, dtype=int)

    times_ms = channel_events["timestamp_ms"].to_numpy(dtype=float)
    states = channel_events["state"].to_numpy(dtype=int)
    first_gt_ms = times_ms[0]

    sample_times_s = np.arange(t_start_s, t_end_s, 1.0)
    sample_times_ms = sample_times_s * 1000.0

    # For each sample time, find last GT event at or before it
    indices = np.searchsorted(times_ms, sample_times_ms, side="right") - 1
    binary = np.full(len(sample_times_s), -1, dtype=int)
    valid = indices >= 0
    # Only assign known state if sample time is at or after first GT event
    known = valid & (sample_times_ms >= first_gt_ms)
    gt_states = states[indices[known]]
    binary[known] = (gt_states == STATE_GREEN).astype(int)
    return sample_times_s, binary


# ---------------------------------------------------------------------------
# Inferred binary state series
# ---------------------------------------------------------------------------


def _inferred_binary_at_seconds(
    windows: pd.DataFrame,
    t_start_s: float,
    t_end_s: float,
    offset_ms: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time_seconds, binary_state) at 1-second resolution.

    binary_state: 1 = green, 0 = red.  Inference has no yellow.
    """
    sample_times_s = np.arange(t_start_s, t_end_s, 1.0)
    binary = np.zeros(len(sample_times_s), dtype=int)  # default red

    if windows.empty:
        return sample_times_s, binary

    for _, row in windows.iterrows():
        gs = _safe_float(row.get("green_start_ms"))
        ge = _safe_float(row.get("green_end_ms"))
        if math.isnan(gs) or math.isnan(ge):
            continue
        # Apply offset: shift inferred windows by -offset to align with GT
        gs_aligned = (gs - offset_ms) / 1000.0
        ge_aligned = (ge - offset_ms) / 1000.0
        mask = (sample_times_s >= gs_aligned) & (sample_times_s < ge_aligned)
        binary[mask] = 1
    return sample_times_s, binary


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def _compute_pair_metrics(
    gt_binary: np.ndarray,
    inferred_binary: np.ndarray,
    gt_cycle_s: float,
    inferred_cycle_s: float,
    gt_green_dur_s: float,
    inferred_green_dur_s: float,
    green_onset_offsets: list[float],
    match_score: float,
) -> dict[str, Any]:
    known = gt_binary >= 0
    if known.sum() == 0:
        return {
            "agreement_rate": float("nan"),
            "green_precision": float("nan"),
            "green_recall": float("nan"),
            "green_f1": float("nan"),
            "cycle_diff_pct": float("nan"),
            "green_duration_diff_pct": float("nan"),
            "median_green_onset_offset_s": float("nan"),
            "match_score": match_score,
        }

    gt_known = gt_binary[known]
    inf_known = inferred_binary[known]

    agree = (gt_known == inf_known).sum()
    agreement_rate = float(agree) / float(len(gt_known))

    inf_green = inf_known == 1
    gt_green = gt_known == 1

    tp = int((gt_green & inf_green).sum())
    fp = int((~gt_green & inf_green).sum())
    fn = int((gt_green & ~inf_green).sum())

    green_precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
    green_recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
    green_f1 = (
        2.0 * green_precision * green_recall / (green_precision + green_recall)
        if (green_precision + green_recall) > 0
        else 0.0
    )

    if not math.isnan(gt_cycle_s) and gt_cycle_s > 0 and not math.isnan(inferred_cycle_s):
        cycle_diff_pct = 100.0 * abs(gt_cycle_s - inferred_cycle_s) / gt_cycle_s
    else:
        cycle_diff_pct = float("nan")

    if not math.isnan(gt_green_dur_s) and gt_green_dur_s > 0 and not math.isnan(inferred_green_dur_s):
        green_duration_diff_pct = 100.0 * abs(gt_green_dur_s - inferred_green_dur_s) / gt_green_dur_s
    else:
        green_duration_diff_pct = float("nan")

    if green_onset_offsets:
        median_green_onset_offset_s = float(np.median(green_onset_offsets))
    else:
        median_green_onset_offset_s = float("nan")

    return {
        "agreement_rate": agreement_rate,
        "green_precision": green_precision,
        "green_recall": green_recall,
        "green_f1": green_f1,
        "cycle_diff_pct": cycle_diff_pct,
        "green_duration_diff_pct": green_duration_diff_pct,
        "median_green_onset_offset_s": median_green_onset_offset_s,
        "match_score": match_score,
    }


def _green_onset_offsets(
    channel_events: pd.DataFrame,
    windows: pd.DataFrame,
    offset_ms: float = 0.0,
) -> list[float]:
    """Compute offsets between inferred green starts and nearest GT green starts."""
    if channel_events.empty or windows.empty:
        return []

    gt_times = channel_events["timestamp_ms"].to_numpy(dtype=float)
    gt_states = channel_events["state"].to_numpy(dtype=int)
    gt_green_starts_ms = gt_times[gt_states == STATE_GREEN]
    if len(gt_green_starts_ms) == 0:
        return []

    offsets: list[float] = []
    for _, row in windows.iterrows():
        gs = _safe_float(row.get("green_start_ms"))
        if math.isnan(gs):
            continue
        # Align inferred start to GT time
        inf_start_ms = gs - offset_ms
        dists = np.abs(gt_green_starts_ms - inf_start_ms) / 1000.0
        offsets.append(float(np.min(dists)))
    return offsets


# ---------------------------------------------------------------------------
# Inferred green/red duration from windows
# ---------------------------------------------------------------------------


def _inferred_green_duration_s(windows: pd.DataFrame) -> float:
    if windows.empty:
        return float("nan")
    durations = []
    for _, row in windows.iterrows():
        d = _safe_float(row.get("green_duration_ms"))
        if not math.isnan(d) and d > 0:
            durations.append(d / 1000.0)
    return float(np.median(durations)) if durations else float("nan")


# ---------------------------------------------------------------------------
# Figure 1: Per-pair timeline comparison
# ---------------------------------------------------------------------------


def _plot_pair_timeline(
    channel_events: pd.DataFrame,
    windows: pd.DataFrame,
    crossing_events: pd.DataFrame,
    pair_row: dict[str, Any],
    channel_cycle_row: dict[str, Any] | None,
    cycle_inference_row: dict[str, Any] | None,
    t_start_s: float,
    t_end_s: float,
    output_path: Path,
) -> dict[str, Any]:
    """Produce Figure 1 for one matched pair. Return metrics dict."""
    city = pair_row["city"]
    record = pair_row["record_name"]
    channel = pair_row["traffic_light_channel"]
    stopline_id = str(pair_row["stopline_id"])
    movement = pair_row["movement"]
    match_score = _safe_float(pair_row.get("match_score", float("nan")))
    offset_ms = _safe_float(pair_row.get("best_offset_ms", 0.0))

    # Compute binary series
    gt_times_s, gt_binary = _gt_binary_at_seconds(channel_events, t_start_s, t_end_s)
    inf_times_s, inf_binary = _inferred_binary_at_seconds(windows, t_start_s, t_end_s, offset_ms)

    # Green onset offsets
    onset_offsets = _green_onset_offsets(channel_events, windows, offset_ms)

    # GT cycle and green duration
    gt_cycle_s = float("nan")
    gt_green_dur_s = float("nan")
    if channel_cycle_row:
        gt_cycle_s = _safe_float(channel_cycle_row.get("estimated_cycle_s"))
        gt_green_dur_s = _safe_float(channel_cycle_row.get("green_duration_s"))

    # Inferred cycle and green duration
    inferred_cycle_s = float("nan")
    inferred_green_dur_s = _inferred_green_duration_s(windows)
    if cycle_inference_row:
        inferred_cycle_s = _safe_float(cycle_inference_row.get("estimated_cycle_s"))

    metrics = _compute_pair_metrics(
        gt_binary, inf_binary,
        gt_cycle_s, inferred_cycle_s,
        gt_green_dur_s, inferred_green_dur_s,
        onset_offsets, match_score,
    )
    metrics["city"] = city
    metrics["record_name"] = record
    metrics["traffic_light_channel"] = channel
    metrics["stopline_id"] = stopline_id
    metrics["movement"] = movement

    # ---- Plot ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 5.5), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1, 0.6]})

    time_ms = channel_events["timestamp_ms"].to_numpy(dtype=float) if not channel_events.empty else np.array([])
    time_s = time_ms / 1000.0
    states = channel_events["state"].to_numpy(dtype=int) if not channel_events.empty else np.array([])

    # Row 1: GT Channel State (full R/G/Y)
    ax1 = axes[0]
    if len(time_s) >= 2:
        start_idx = 0
        for idx in range(1, len(time_s)):
            if states[idx] != states[start_idx]:
                ax1.fill_between(
                    [time_s[start_idx], time_s[idx]],
                    [0, 0], [1, 1],
                    color=STATE_COLOR.get(int(states[start_idx]), COLOR_UNKNOWN),
                )
                start_idx = idx
        ax1.fill_between(
            [time_s[start_idx], time_s[-1]],
            [0, 0], [1, 1],
            color=STATE_COLOR.get(int(states[start_idx]), COLOR_UNKNOWN),
        )
    # Crossing event ticks
    if not crossing_events.empty:
        cross_ts = crossing_events["crossing_timestamp_ms"].to_numpy(dtype=float) / 1000.0
        in_range = (cross_ts >= t_start_s) & (cross_ts <= t_end_s)
        ax1.vlines(cross_ts[in_range], 0, 1, colors="white", linewidth=0.4, alpha=0.7)
    ax1.set_ylabel("GT State")
    ax1.set_yticks([0.5])
    ax1.set_yticklabels([""])
    ax1.set_ylim(0, 1)

    # Row 2: Inferred Signal Windows
    ax2 = axes[1]
    if not windows.empty:
        for _, row in windows.iterrows():
            gs = _safe_float(row.get("green_start_ms"))
            ge = _safe_float(row.get("green_end_ms"))
            re_end = _safe_float(row.get("red_end_ms"))
            if math.isnan(gs):
                continue
            gs_s = (gs - offset_ms) / 1000.0
            ge_s = (ge - offset_ms) / 1000.0
            if not math.isnan(ge) and ge > gs:
                ax2.fill_between([gs_s, ge_s], [0, 0], [1, 1], color=STATE_COLOR[STATE_GREEN])
            # Red window: from green_end to red_end or next green_start
            if not math.isnan(re_end) and re_end > ge:
                re_s = (re_end - offset_ms) / 1000.0
                ax2.fill_between([ge_s, re_s], [0, 0], [1, 1], color=STATE_COLOR[STATE_RED])
            elif math.isnan(re_end) and not math.isnan(ge):
                # Extend red to t_end
                ax2.fill_between([ge_s, t_end_s], [0, 0], [1, 1], color=STATE_COLOR[STATE_RED], alpha=0.5)
    ax2.set_ylabel("Inferred")
    ax2.set_yticks([0.5])
    ax2.set_yticklabels([""])
    ax2.set_ylim(0, 1)

    # Row 3: Agreement Raster
    ax3 = axes[2]
    agreement = np.full(len(gt_times_s), -1, dtype=int)
    # 0=agree, 1=FN (GT=G, inf=red), 2=FP (GT≠G, inf=green), -1=unknown
    known = gt_binary >= 0
    gt_g = gt_binary == 1
    inf_g = inf_binary == 1
    agreement[known & (gt_binary == inf_binary)] = 0   # agree
    agreement[known & gt_g & ~inf_g] = 1                # FN
    agreement[known & ~gt_g & inf_g] = 2                # FP
    agreement[~known] = -1                               # unknown

    colors_raster = np.empty(len(gt_times_s), dtype=object)
    colors_raster[agreement == 0] = COLOR_AGREE
    colors_raster[agreement == 1] = COLOR_FN
    colors_raster[agreement == 2] = COLOR_FP
    colors_raster[agreement == -1] = COLOR_UNKNOWN

    for i in range(len(gt_times_s)):
        ax3.fill_between(
            [gt_times_s[i], gt_times_s[i] + 1],
            [0, 0], [1, 1],
            color=colors_raster[i],
        )
    ax3.set_ylabel("Agreement")
    ax3.set_yticks([0.5])
    ax3.set_yticklabels([""])
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("Time (s)")

    for ax in axes:
        ax.set_xlim(t_start_s, t_end_s)

    title = f"{city} | {record} | {channel} → stopline {stopline_id} {movement} | score={match_score:.2f}"
    fig.suptitle(title, fontsize=10, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return metrics


# ---------------------------------------------------------------------------
# Figure 2: City overview grid
# ---------------------------------------------------------------------------


def _plot_city_grid(
    city: str,
    pair_data: list[dict[str, Any]],
    t_start_s: float,
    t_end_s: float,
    output_path: Path,
) -> None:
    """Produce Figure 2: stack of all matched pairs for a city."""
    if not pair_data:
        return

    n_pairs = len(pair_data)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(14, 2.2 * n_pairs), sharex=True)
    if n_pairs == 1:
        axes = [axes]

    for idx, pd_item in enumerate(pair_data):
        ax = axes[idx]
        channel_events = pd_item["channel_events"]
        windows = pd_item["windows"]
        crossing_events = pd_item["crossing_events"]
        pair_row = pd_item["pair_row"]
        offset_ms = _safe_float(pair_row.get("best_offset_ms", 0.0))

        # Three sub-rows: GT, Inferred, Agreement
        sub_y = [0.67, 0.33, 0.0]
        sub_h = 0.28

        # GT
        time_ms = channel_events["timestamp_ms"].to_numpy(dtype=float) if not channel_events.empty else np.array([])
        time_s = time_ms / 1000.0
        states = channel_events["state"].to_numpy(dtype=int) if not channel_events.empty else np.array([])
        if len(time_s) >= 2:
            start_idx = 0
            for si in range(1, len(time_s)):
                if states[si] != states[start_idx]:
                    ax.fill_between(
                        [time_s[start_idx], time_s[si]],
                        [sub_y[0]] * 2, [sub_y[0] + sub_h] * 2,
                        color=STATE_COLOR.get(int(states[start_idx]), COLOR_UNKNOWN),
                    )
                    start_idx = si
            ax.fill_between(
                [time_s[start_idx], time_s[-1]],
                [sub_y[0]] * 2, [sub_y[0] + sub_h] * 2,
                color=STATE_COLOR.get(int(states[start_idx]), COLOR_UNKNOWN),
            )
        if not crossing_events.empty:
            cross_ts = crossing_events["crossing_timestamp_ms"].to_numpy(dtype=float) / 1000.0
            in_range = (cross_ts >= t_start_s) & (cross_ts <= t_end_s)
            ax.vlines(cross_ts[in_range], sub_y[0], sub_y[0] + sub_h,
                       colors="white", linewidth=0.3, alpha=0.6)

        # Inferred
        if not windows.empty:
            for _, row in windows.iterrows():
                gs = _safe_float(row.get("green_start_ms"))
                ge = _safe_float(row.get("green_end_ms"))
                re_end = _safe_float(row.get("red_end_ms"))
                if math.isnan(gs):
                    continue
                gs_s = (gs - offset_ms) / 1000.0
                ge_s = (ge - offset_ms) / 1000.0
                if not math.isnan(ge) and ge > gs:
                    ax.fill_between([gs_s, ge_s],
                                    [sub_y[1]] * 2, [sub_y[1] + sub_h] * 2,
                                    color=STATE_COLOR[STATE_GREEN])
                if not math.isnan(re_end) and re_end > ge:
                    re_s = (re_end - offset_ms) / 1000.0
                    ax.fill_between([ge_s, re_s],
                                    [sub_y[1]] * 2, [sub_y[1] + sub_h] * 2,
                                    color=STATE_COLOR[STATE_RED])

        # Agreement raster
        gt_times_s, gt_binary = _gt_binary_at_seconds(channel_events, t_start_s, t_end_s)
        _, inf_binary = _inferred_binary_at_seconds(windows, t_start_s, t_end_s, offset_ms)
        agreement = np.full(len(gt_times_s), -1, dtype=int)
        known = gt_binary >= 0
        agreement[known & (gt_binary == inf_binary)] = 0
        agreement[known & (gt_binary == 1) & (inf_binary == 0)] = 1
        agreement[known & (gt_binary == 0) & (inf_binary == 1)] = 2
        agreement[~known] = -1

        raster_colors = {0: COLOR_AGREE, 1: COLOR_FN, 2: COLOR_FP, -1: COLOR_UNKNOWN}
        for i in range(0, len(gt_times_s), 2):  # sample every 2s to reduce draw calls
            end_i = min(i + 2, len(gt_times_s))
            c = raster_colors.get(agreement[i], COLOR_UNKNOWN)
            ax.fill_between(
                [gt_times_s[i], gt_times_s[min(end_i - 1 + 1, len(gt_times_s) - 1)] + 1],
                [sub_y[2]] * 2, [sub_y[2] + sub_h] * 2,
                color=c,
            )

        ax.set_xlim(t_start_s, t_end_s)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        channel = pair_row["traffic_light_channel"]
        stopline_id = str(pair_row["stopline_id"])
        movement = pair_row["movement"]
        score = _safe_float(pair_row.get("match_score", 0.0))
        ax.set_ylabel(f"{channel}\n→ {stopline_id} {movement}\nscore={score:.2f}",
                       fontsize=7)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{city} — Signal Inference vs GT Comparison Grid", fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Cycle comparison
# ---------------------------------------------------------------------------


def _plot_cycle_comparison(
    city: str,
    metrics_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Grouped bar chart (left) and scatter (right) for cycle comparison."""
    if not metrics_rows:
        return

    df = pd.DataFrame(metrics_rows)
    gt_cycles = df["gt_cycle_s"].to_numpy(dtype=float)
    inf_cycles = df["inferred_cycle_s"].to_numpy(dtype=float)
    labels = [f"{r['stopline_id']}\n{r['movement']}" for r in metrics_rows]

    # Filter out NaN pairs
    valid = np.isfinite(gt_cycles) & np.isfinite(inf_cycles)
    if valid.sum() == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                     gridspec_kw={"width_ratios": [1.6, 1.0]})

    # Left: grouped bar
    x = np.arange(valid.sum())
    w = 0.35
    ax1.bar(x - w / 2, gt_cycles[valid], w, label="GT Cycle", color=STATE_COLOR[STATE_RED])
    ax1.bar(x + w / 2, inf_cycles[valid], w, label="Inferred Cycle", color=STATE_COLOR[STATE_GREEN])
    valid_labels = [l for l, v in zip(labels, valid) if v]
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_labels, fontsize=7, rotation=45, ha="right")
    ax1.set_ylabel("Cycle Length (s)")
    ax1.set_title(f"{city} — GT vs Inferred Cycle")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.25)

    # Right: scatter
    ax2.scatter(gt_cycles[valid], inf_cycles[valid], c=STATE_COLOR[STATE_GREEN],
                edgecolors="black", linewidth=0.5, s=40, zorder=3)
    lo = min(gt_cycles[valid].min(), inf_cycles[valid].min()) - 5
    hi = max(gt_cycles[valid].max(), inf_cycles[valid].max()) + 5
    ax2.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, alpha=0.5, label="y = x")
    ax2.set_xlabel("GT Cycle (s)")
    ax2.set_ylabel("Inferred Cycle (s)")
    ax2.set_title(f"{city} — Cycle Correlation")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)
    ax2.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Offset distribution histogram
# ---------------------------------------------------------------------------


def _plot_offset_distribution(
    all_offsets: list[float],
    output_path: Path,
) -> None:
    if not all_offsets:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    offsets = np.array(all_offsets)
    ax.hist(offsets / 1000.0, bins=40, color=STATE_COLOR[STATE_GREEN],
            edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("best_offset (s)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of best_offset_ms Across Matched Pairs")
    ax.grid(axis="y", alpha=0.25)
    median_offset = float(np.median(offsets)) / 1000.0
    ax.axvline(median_offset, color=STATE_COLOR[STATE_RED], linestyle="-", linewidth=1.2,
               label=f"median = {median_offset:.1f}s")
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _run_city(
    city: str,
    records: list[RecordDescription],
    mapping: pd.DataFrame,
    phase_events: pd.DataFrame,
    channel_cycles: pd.DataFrame,
    windows_df: pd.DataFrame,
    cycle_inf_df: pd.DataFrame,
    crossings_df: pd.DataFrame,
    figure_root: Path,
    time_window_s: float,
    top_n: int,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Process one city. Return (metrics_rows, offsets_ms)."""
    city_map = mapping[mapping["city"] == city].copy()
    if city_map.empty:
        print(f"[skip] {city}: no matched pairs")
        return [], []

    offsets_ms: list[float] = []
    metrics_rows: list[dict[str, Any]] = []
    pair_data: list[dict[str, Any]] = []

    # Sort by match_score descending for top-N
    city_map = city_map.sort_values("match_score", ascending=False)

    for _, pair_row in city_map.iterrows():
        record_name = pair_row["record_name"]
        channel = pair_row["traffic_light_channel"]
        stopline_id = str(pair_row["stopline_id"])
        movement = pair_row["movement"]

        # Skip non-vehicle channels
        if pair_row.get("channel_type", "vehicle") != "vehicle":
            continue

        offset_ms = _safe_float(pair_row.get("best_offset_ms", 0.0))
        offsets_ms.append(offset_ms)

        # Load GT channel events for this pair
        record_obj = None
        for r in records:
            if r.record_name == record_name:
                record_obj = r
                break
        if record_obj is None:
            continue

        gt_channels = load_traffic_light_channels(record_obj)
        channel_events = gt_channels[gt_channels["traffic_light_channel"] == channel].copy()
        if channel_events.empty:
            continue

        # Determine time window from GT data
        gt_time_ms = channel_events["timestamp_ms"].to_numpy(dtype=float)
        if len(gt_time_ms) == 0:
            continue
        t_start_s = float(gt_time_ms[0]) / 1000.0
        t_end_s = min(t_start_s + time_window_s, float(gt_time_ms[-1]) / 1000.0)

        # Inferred windows for this (stopline, movement)
        pair_windows = windows_df[
            (windows_df["city"] == city) &
            (windows_df["record_name"] == record_name) &
            (windows_df["stopline_id"].astype(str) == stopline_id) &
            (windows_df["movement"] == movement)
        ].copy()

        # Crossing events for overlay
        pair_crossings = crossings_df[
            (crossings_df["city"] == city) &
            (crossings_df["record_name"] == record_name) &
            (crossings_df["stopline_id"].astype(str) == stopline_id) &
            (crossings_df["movement"] == movement)
        ].copy()

        # GT channel cycle info
        ch_cycle_row = None
        ch_cycles = channel_cycles[
            (channel_cycles["city"] == city) &
            (channel_cycles["record_name"] == record_name) &
            (channel_cycles["traffic_light_channel"] == channel)
        ]
        if not ch_cycles.empty:
            ch_cycle_row = ch_cycles.iloc[0].to_dict()

        # Inferred cycle info
        cy_inf_row = None
        cy_infs = cycle_inf_df[
            (cycle_inf_df["city"] == city) &
            (cycle_inf_df["record_name"] == record_name) &
            (cycle_inf_df["stopline_id"].astype(str) == stopline_id) &
            (cycle_inf_df["movement"] == movement)
        ]
        if not cy_infs.empty:
            cy_inf_row = cy_infs.iloc[0].to_dict()

        # Per-pair timeline figure
        fig_name = f"{record_name}__{channel.replace(' ', '_')}__stopline_{stopline_id}_{movement}.png"
        fig_path = figure_root / city / fig_name
        m = _plot_pair_timeline(
            channel_events, pair_windows, pair_crossings,
            pair_row.to_dict(), ch_cycle_row, cy_inf_row,
            t_start_s, t_end_s, fig_path,
        )
        # Attach extra metric fields
        m["gt_cycle_s"] = _safe_float(ch_cycle_row.get("estimated_cycle_s")) if ch_cycle_row else float("nan")
        m["inferred_cycle_s"] = _safe_float(cy_inf_row.get("estimated_cycle_s")) if cy_inf_row else float("nan")
        m["gt_green_duration_s"] = _safe_float(ch_cycle_row.get("green_duration_s")) if ch_cycle_row else float("nan")
        m["inferred_green_duration_s"] = _inferred_green_duration_s(pair_windows)
        m["best_offset_ms"] = offset_ms
        m["confidence"] = pair_row.get("confidence", "")
        metrics_rows.append(m)

        pair_data.append({
            "channel_events": channel_events,
            "windows": pair_windows,
            "crossing_events": pair_crossings,
            "pair_row": pair_row.to_dict(),
        })

    # City overview grid (top-N pairs)
    top_pairs = pair_data[:top_n]
    # Use time window from first pair
    if top_pairs:
        first_ce = top_pairs[0]["channel_events"]
        if not first_ce.empty:
            t0 = float(first_ce["timestamp_ms"].iloc[0]) / 1000.0
            t1 = min(t0 + time_window_s, float(first_ce["timestamp_ms"].iloc[-1]) / 1000.0)
        else:
            t0, t1 = 0.0, time_window_s
    else:
        t0, t1 = 0.0, time_window_s

    _plot_city_grid(city, top_pairs, t0, t1,
                    figure_root / city / f"{city.lower()}_comparison_grid.png")

    # Cycle comparison
    _plot_cycle_comparison(city, metrics_rows,
                           figure_root / city / f"{city.lower()}_cycle_comparison.png")

    return metrics_rows, offsets_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare inferred signal states against GT traffic-light CSV data."
    )
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "SinD" / "Dataset")
    parser.add_argument("--map-fallback-root", type=Path, default=PROJECT_ROOT / "SinD" / "Data")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output" / "lights")
    parser.add_argument("--figure-root", type=Path, default=None,
                        help="Where to save figures (default: output-root/comparison_figures)")
    parser.add_argument("--cities", nargs="*", default=None,
                        help="Which cities to process (default: all 4)")
    parser.add_argument("--records", nargs="*", default=None,
                        help="Specific record names; if omitted, pick best per city")
    parser.add_argument("--all-records", action="store_true",
                        help="Produce comparison for all records, not just representative")
    parser.add_argument("--time-window-s", type=float, default=600,
                        help="Seconds of timeline to show (default: 600)")
    parser.add_argument("--top-n", type=int, default=8,
                        help="Max pairs per city in grid figure (default: 8)")
    args = parser.parse_args()

    output_root = args.output_root
    figure_root = args.figure_root or output_root / "comparison_figures"

    # Load pre-computed artifacts
    mapping = _load_csv(output_root / "traffic_light_channel_mapping.csv")
    phase_events = _load_csv(output_root / "traffic_light_channel_phase_events.csv")
    channel_cycles = _load_csv(output_root / "traffic_light_channel_cycle.csv")
    windows_df = _load_csv(output_root / "movement_signal_windows.csv")
    cycle_inf_df = _load_csv(output_root / "cycle_inference.csv")
    crossings_df = _load_csv(output_root / "crossing_events.csv")

    if mapping.empty:
        raise SystemExit(f"No traffic_light_channel_mapping.csv found in {output_root}")

    # Discover records for GT loading
    all_records = discover_records(args.data_root, args.map_fallback_root)

    # Determine cities
    if args.cities:
        cities = args.cities
    else:
        cities = sorted(mapping["city"].unique().tolist())

    all_metrics: list[dict[str, Any]] = []
    all_offsets_ms: list[float] = []

    for city in cities:
        city_records = _records_for_city(
            all_records, city, args.records, args.all_records, mapping
        )
        if not city_records:
            print(f"[skip] {city}: no records found")
            continue

        print(f"[info] {city}: processing {len(city_records)} record(s)")
        metrics, offsets = _run_city(
            city, city_records, mapping, phase_events, channel_cycles,
            windows_df, cycle_inf_df, crossings_df,
            figure_root, args.time_window_s, args.top_n,
        )
        all_metrics.extend(metrics)
        all_offsets_ms.extend(offsets)
        print(f"[done] {city}: {len(metrics)} pairs compared")

    # Global offset distribution
    _plot_offset_distribution(all_offsets_ms, figure_root / "offset_distribution.png")

    # Write metrics CSVs
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        # Reorder columns
        key_cols = ["city", "record_name", "traffic_light_channel", "stopline_id",
                     "movement", "confidence", "match_score"]
        other_cols = [c for c in metrics_df.columns if c not in key_cols]
        metrics_df = metrics_df[key_cols + other_cols]
        metrics_df.to_csv(figure_root / "signal_comparison_metrics.csv", index=False)

        # City summary
        summary_rows = []
        for city in cities:
            city_m = metrics_df[metrics_df["city"] == city]
            if city_m.empty:
                continue
            summary_rows.append({
                "city": city,
                "num_pairs": len(city_m),
                "mean_agreement_rate": city_m["agreement_rate"].mean(),
                "mean_green_precision": city_m["green_precision"].mean(),
                "mean_green_recall": city_m["green_recall"].mean(),
                "mean_green_f1": city_m["green_f1"].mean(),
                "mean_cycle_diff_pct": city_m["cycle_diff_pct"].mean(),
                "mean_green_duration_diff_pct": city_m["green_duration_diff_pct"].mean(),
                "median_green_onset_offset_s": city_m["median_green_onset_offset_s"].median(),
                "median_best_offset_ms": float(np.median([
                    _safe_float(v) for v in city_m["best_offset_ms"].tolist()
                ])) if not city_m.empty else float("nan"),
            })
        pd.DataFrame(summary_rows).to_csv(figure_root / "city_summary_metrics.csv", index=False)

        # README
        _write_readme(figure_root, cities, all_metrics, all_offsets_ms)

    print(f"[done] all figures and metrics saved to {figure_root}")


def _write_readme(
    figure_root: Path,
    cities: list[str],
    all_metrics: list[dict[str, Any]],
    all_offsets_ms: list[float],
) -> None:
    lines = [
        "# Signal Inference vs Ground-Truth Comparison",
        "",
        "## Overview",
        "",
        f"- Cities: {', '.join(cities)}",
        f"- Total matched pairs: {len(all_metrics)}",
    ]

    metrics_df = pd.DataFrame(all_metrics)
    if not metrics_df.empty:
        lines += [
            f"- Mean agreement rate: {metrics_df['agreement_rate'].mean():.3f}",
            f"- Mean green F1: {metrics_df['green_f1'].mean():.3f}",
            f"- Mean green precision: {metrics_df['green_precision'].mean():.3f}",
            f"- Mean green recall: {metrics_df['green_recall'].mean():.3f}",
        ]

    if all_offsets_ms:
        offsets = np.array(all_offsets_ms)
        lines.append(f"- Median best_offset_ms: {float(np.median(offsets)):.0f} ms ({float(np.median(offsets))/1000:.1f} s)")

    lines += [
        "",
        "## Notes",
        "",
        "- GT is reduced to binary (green vs non-green) for metric computation, as inference only detects green/red.",
        "- Right turns may have poor green recall because vehicles often turn right on red; such pairs are included but metrics should be interpreted accordingly.",
        "- Yellow states in GT are counted as non-green in binary comparison.",
        "- `best_offset_ms` from the channel mapping pipeline is applied to align inferred windows with GT before comparison.",
        "",
        "## Output Files",
        "",
        "```\n{City}/\n    {record}__{channel}__stopline_{id}_{movement}.png   # per-pair timeline\n    {city}_comparison_grid.png                           # city overview\n    {city}_cycle_comparison.png                          # cycle bar+scatter\noffset_distribution.png                                  # global offset histogram\nsignal_comparison_metrics.csv                            # per-pair metrics\ncity_summary_metrics.csv                                 # per-city aggregates\nREADME.md                                                # this file\n```",
        "",
        "## Figure Descriptions",
        "",
        "### Figure 1: Per-Pair Timeline Comparison",
        "3-row panel per matched pair:",
        "- Row 1: GT channel state (R/G/Y) with crossing event tick marks",
        "- Row 2: Inferred green/red signal windows",
        "- Row 3: Agreement raster (green=agree, orange=false negative, purple=false positive, grey=unknown)",
        "",
        "### Figure 2: City Overview Grid",
        "Stack of top-N pairs (3 sub-rows each) for a city, sorted by match_score.",
        "",
        "### Figure 3: Cycle Comparison",
        "Left: grouped bar chart of GT vs inferred cycle per pair. Right: scatter with diagonal reference.",
        "",
        "### Figure 4: Offset Distribution",
        "Histogram of best_offset_ms across all matched pairs.",
    ]

    (figure_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
