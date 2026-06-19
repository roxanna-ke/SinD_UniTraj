from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sind_converter.data.discovery import discover_records
from sind_converter.lights.channel_matching import (
    AUDIT_FIELDS,
    CHANNEL_CYCLE_FIELDS,
    FINAL_MAPPING_FIELDS,
    MatchingConfig,
    PHASE_EVENT_FIELDS,
    infer_channel_cycles,
    match_record_channels,
    phase_event_rows,
)
from sind_converter.lights.infer_stopline_signals import (
    CROSSING_EVENT_FIELDS,
    CYCLE_INFERENCE_FIELDS,
    MOVEMENT_BURST_FIELDS,
    MOVEMENT_SIGNAL_WINDOW_FIELDS,
    InferenceConfig,
    infer_crossing_events,
    infer_cycles,
    infer_movement_bursts,
    infer_signal_windows,
)
from sind_converter.lights.stopline_extraction import (
    STOP_LINE_FIELDS,
    extract_stop_lines,
    extract_traffic_light_relations,
)


RELATION_FIELDS = [
    "city",
    "osm_relation_id",
    "osm_relation_expected_stopline",
    "osm_relation_refers_way",
    "osm_relation_expected_channel",
]

GENERATED_ARTIFACTS = (
    "stop_lines.csv",
    "osm_traffic_light_relations.csv",
    "crossing_events.csv",
    "movement_burst.csv",
    "cycle_inference.csv",
    "movement_signal_windows.csv",
    "traffic_light_channel_phase_events.csv",
    "traffic_light_channel_cycle.csv",
    "traffic_light_mapping_audit.csv",
    "traffic_light_channel_mapping.csv",
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _records_by_city_map(records):
    city_maps = {}
    for record in records:
        city_maps.setdefault(record.city, record.map_path)
    return city_maps


def _clear_output_artifacts(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for filename in GENERATED_ARTIFACTS:
        path = output_root / filename
        if path.exists():
            path.unlink()


def _warn_stale_right_turn_configs(project_root: Path) -> None:
    config_root = project_root / "sind_converter" / "lights" / "config"
    if not config_root.exists():
        return
    stale_paths: list[Path] = []
    for path in sorted(config_root.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if '"movement": "right"' in json.dumps(payload, ensure_ascii=True, sort_keys=True):
            stale_paths.append(path)
    if stale_paths:
        print("[warn] existing traffic-light config files still contain movement=right and are stale under the new policy:")
        for path in stale_paths:
            print(f"  - {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer and match SinD traffic-light channels to stop-line movements.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--map-fallback-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--segment-tolerance-m", type=float, default=2.0)
    parser.add_argument("--min-crossing-speed-mps", type=float, default=0.5)
    parser.add_argument("--dedupe-window-ms", type=float, default=1000.0)
    parser.add_argument("--heading-turn-threshold-deg", type=float, default=35.0)
    parser.add_argument("--min-track-frames", type=int, default=15)
    parser.add_argument("--min-track-duration-ms", type=float, default=1500.0)
    parser.add_argument("--max-frame-gap-ms", type=float, default=300.0)
    parser.add_argument("--min-crossing-context-frames", type=int, default=5)
    parser.add_argument("--min-pre-crossing-displacement-m", type=float, default=3.0)
    parser.add_argument("--min-post-crossing-displacement-m", type=float, default=3.0)
    parser.add_argument("--burst-gap-s", type=float, default=4.0)
    parser.add_argument("--min-cycle-s", type=float, default=60.0)
    parser.add_argument("--max-cycle-s", type=float, default=240.0)
    parser.add_argument("--min-support-crossings", type=int, default=2)
    parser.add_argument("--high-score-threshold", type=float, default=0.75)
    parser.add_argument("--medium-score-threshold", type=float, default=0.55)
    parser.add_argument("--offset-search-ms", type=int, default=20_000)
    parser.add_argument("--offset-step-ms", type=int, default=500)
    parser.add_argument("--max-cycle-diff-s", type=float, default=15.0)
    args = parser.parse_args()

    map_fallback_root = args.map_fallback_root or args.data_root
    records = discover_records(args.data_root, map_fallback_root, cities=args.cities)
    if args.max_records is not None:
        records = records[: args.max_records]
    if not records:
        raise SystemExit("No SinD records discovered.")

    output_root = args.output_root
    _clear_output_artifacts(output_root)
    _warn_stale_right_turn_configs(Path(__file__).resolve().parents[2])

    inference_cfg = InferenceConfig(
        segment_tolerance_m=args.segment_tolerance_m,
        min_crossing_speed_mps=args.min_crossing_speed_mps,
        dedupe_window_ms=args.dedupe_window_ms,
        heading_turn_threshold_deg=args.heading_turn_threshold_deg,
        min_track_frames=args.min_track_frames,
        min_track_duration_ms=args.min_track_duration_ms,
        max_frame_gap_ms=args.max_frame_gap_ms,
        min_crossing_context_frames=args.min_crossing_context_frames,
        min_pre_crossing_displacement_m=args.min_pre_crossing_displacement_m,
        min_post_crossing_displacement_m=args.min_post_crossing_displacement_m,
        burst_gap_s=args.burst_gap_s,
        min_cycle_s=args.min_cycle_s,
        max_cycle_s=args.max_cycle_s,
    )
    matching_cfg = MatchingConfig(
        min_support_crossings=args.min_support_crossings,
        high_score_threshold=args.high_score_threshold,
        medium_score_threshold=args.medium_score_threshold,
        offset_search_ms=args.offset_search_ms,
        offset_step_ms=args.offset_step_ms,
        max_cycle_diff_s=args.max_cycle_diff_s,
    )

    stop_lines_by_city = {}
    stop_line_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, str]] = []
    for city, map_path in sorted(_records_by_city_map(records).items()):
        stop_lines = extract_stop_lines(map_path, city)
        stop_lines_by_city[city] = stop_lines
        stop_line_rows.extend([line.as_row() for line in stop_lines])
        relation_rows.extend(extract_traffic_light_relations(map_path, city))
        print(f"[info] {city}: {len(stop_lines)} stop lines from {map_path}")

    _write_csv(output_root / "stop_lines.csv", STOP_LINE_FIELDS, stop_line_rows)
    _write_csv(output_root / "osm_traffic_light_relations.csv", RELATION_FIELDS, relation_rows)

    crossing_rows: list[dict[str, Any]] = []
    crossings_by_record: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for idx, record in enumerate(records, start=1):
        rows = infer_crossing_events(record, stop_lines_by_city.get(record.city, []), inference_cfg)
        crossing_rows.extend(rows)
        crossings_by_record[(record.city, record.record_name)] = rows
        print(f"[info] record {idx}/{len(records)} {record.city}/{record.record_name}: {len(rows)} crossing events")

    _write_csv(output_root / "crossing_events.csv", CROSSING_EVENT_FIELDS, crossing_rows)

    burst_rows = infer_movement_bursts(crossing_rows, inference_cfg)
    cycle_rows = infer_cycles(burst_rows, inference_cfg)
    window_rows = infer_signal_windows(burst_rows, cycle_rows, inference_cfg)
    _write_csv(output_root / "movement_burst.csv", MOVEMENT_BURST_FIELDS, burst_rows)
    _write_csv(output_root / "cycle_inference.csv", CYCLE_INFERENCE_FIELDS, cycle_rows)
    _write_csv(output_root / "movement_signal_windows.csv", MOVEMENT_SIGNAL_WINDOW_FIELDS, window_rows)

    phase_rows: list[dict[str, Any]] = []
    channel_cycle_rows: list[dict[str, Any]] = []
    for record in records:
        phase_rows.extend(phase_event_rows(record))
        channel_cycle_rows.extend(infer_channel_cycles(record))
    _write_csv(output_root / "traffic_light_channel_phase_events.csv", PHASE_EVENT_FIELDS, phase_rows)
    _write_csv(output_root / "traffic_light_channel_cycle.csv", CHANNEL_CYCLE_FIELDS, channel_cycle_rows)

    windows_by_record: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in window_rows:
        windows_by_record.setdefault((row["city"], row["record_name"]), []).append(row)

    audit_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    for record in records:
        key = (record.city, record.record_name)
        record_audit, record_final = match_record_channels(
            record,
            crossings_by_record.get(key, []),
            windows_by_record.get(key, []),
            cycle_rows,
            channel_cycle_rows,
            relation_rows,
            matching_cfg,
        )
        audit_rows.extend(record_audit)
        final_rows.extend(record_final)
        print(f"[info] matched {record.city}/{record.record_name}: audit={len(record_audit)} final={len(record_final)}")

    _write_csv(output_root / "traffic_light_mapping_audit.csv", AUDIT_FIELDS, audit_rows)
    _write_csv(output_root / "traffic_light_channel_mapping.csv", FINAL_MAPPING_FIELDS, final_rows)
    print(f"[done] wrote traffic-light inference artifacts to {output_root}")


if __name__ == "__main__":
    main()
