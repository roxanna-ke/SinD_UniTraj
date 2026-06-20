#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT,):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from metadrive.type import MetaDriveType

from sind_converter.config.defaults import ConverterConfig
from sind_converter.data.discovery import discover_records
from sind_converter.lights.runtime_signal_states import dynamic_map_states_for_window
from sind_converter.maps.osm import build_admission_rules, load_training_mapping_table
from sind_converter.scenarios.convert import windows_for_record


TRAFFIC_LIGHT_STATE_TO_INT = {
    None: 0,
    MetaDriveType.LANE_STATE_UNKNOWN: 0,
    MetaDriveType.LANE_STATE_ARROW_STOP: 1,
    MetaDriveType.LANE_STATE_ARROW_CAUTION: 2,
    MetaDriveType.LANE_STATE_ARROW_GO: 3,
    MetaDriveType.LANE_STATE_STOP: 4,
    MetaDriveType.LANE_STATE_CAUTION: 5,
    MetaDriveType.LANE_STATE_GO: 6,
    MetaDriveType.LANE_STATE_FLASHING_STOP: 7,
    MetaDriveType.LANE_STATE_FLASHING_CAUTION: 8,
}


@dataclass
class CoverageCounts:
    records: int = 0
    scenarios: int = 0
    target_samples: int = 0
    scenarios_with_signal_objects: int = 0
    scenarios_with_known_state_any_frame: int = 0
    scenarios_informative_at_center: int = 0
    target_samples_with_signal_objects: int = 0
    target_samples_with_known_state_any_frame: int = 0
    target_samples_informative_at_center: int = 0


def _count_dynamic_map_states(dynamic_map_states: dict[str, dict], current_time_index: int) -> tuple[bool, bool, bool]:
    if not dynamic_map_states:
        return False, False, False
    has_known_any = False
    informative_center = False

    for signal in dynamic_map_states.values():
        states = list(signal.get("state", {}).get("object_state", []))
        if not states:
            continue
        state_ids = [int(TRAFFIC_LIGHT_STATE_TO_INT.get(state, 0)) for state in states]
        if any(state_id > 0 for state_id in state_ids):
            has_known_any = True
        center_idx = min(max(current_time_index, 0), len(state_ids) - 1)
        if state_ids[center_idx] > 0:
            informative_center = True

    return True, has_known_any, informative_center


def _accumulate(counts: CoverageCounts, target_count: int, has_signal_objects: bool, has_known_any: bool, informative_center: bool) -> None:
    counts.scenarios += 1
    counts.target_samples += target_count
    if has_signal_objects:
        counts.scenarios_with_signal_objects += 1
        counts.target_samples_with_signal_objects += target_count
    if has_known_any:
        counts.scenarios_with_known_state_any_frame += 1
        counts.target_samples_with_known_state_any_frame += target_count
    if informative_center:
        counts.scenarios_informative_at_center += 1
        counts.target_samples_informative_at_center += target_count


def _ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _summary_row(scope: str, name: str, counts: CoverageCounts) -> dict[str, object]:
    row = {"scope": scope, "name": name, **asdict(counts)}
    row.update(
        {
            "scenario_signal_object_coverage": _ratio(counts.scenarios_with_signal_objects, counts.scenarios),
            "scenario_known_state_any_frame_coverage": _ratio(counts.scenarios_with_known_state_any_frame, counts.scenarios),
            "scenario_informative_center_coverage": _ratio(counts.scenarios_informative_at_center, counts.scenarios),
            "sample_signal_object_coverage": _ratio(counts.target_samples_with_signal_objects, counts.target_samples),
            "sample_known_state_any_frame_coverage": _ratio(counts.target_samples_with_known_state_any_frame, counts.target_samples),
            "sample_informative_center_coverage": _ratio(counts.target_samples_informative_at_center, counts.target_samples),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze SinD traffic-light coverage directly from raw data using the canonical scenario generation logic."
    )
    parser.add_argument("--data-root", type=Path, default=Path("SinD/Dataset"))
    parser.add_argument("--map-fallback-root", type=Path, default=Path("SinD/Data"))
    parser.add_argument(
        "--training-mapping-table",
        type=Path,
        default=None,
        help="Optional OSM map admission table for matching formal conversion scope; not used for traffic-light binding.",
    )
    parser.add_argument(
        "--traffic-light-binding-root",
        type=Path,
        default=None,
        help="Traffic-light binding config root. Defaults to sind_converter/lights/config.",
    )
    parser.add_argument("--cities", nargs="*", default=None)
    parser.add_argument("--past-len", type=int, default=21)
    parser.add_argument("--future-len", type=int, default=60)
    parser.add_argument("--stride", type=int, default=40)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-scenarios-per-record", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    cfg = ConverterConfig(
        data_root=args.data_root,
        map_fallback_root=args.map_fallback_root,
        canonical_scenario_root=Path("/tmp/unused_canonical"),
        split_root=Path("/tmp/unused_split"),
        cache_root=Path("/tmp/unused_cache"),
        training_mapping_table=args.training_mapping_table,
        traffic_light_binding_root=args.traffic_light_binding_root
        if args.traffic_light_binding_root is not None
        else ConverterConfig(
            data_root=args.data_root,
            map_fallback_root=args.map_fallback_root,
            canonical_scenario_root=Path("/tmp/unused_canonical"),
            split_root=Path("/tmp/unused_split"),
            cache_root=Path("/tmp/unused_cache"),
        ).traffic_light_binding_root,
        past_len=args.past_len,
        future_len=args.future_len,
        stride=args.stride,
    )

    records = discover_records(cfg.data_root, cfg.map_fallback_root, cities=args.cities)
    if args.max_records is not None:
        records = records[: args.max_records]

    admission_rules = (
        build_admission_rules(load_training_mapping_table(cfg.training_mapping_table))
        if cfg.training_mapping_table is not None
        else None
    )
    signal_binding_source = str(cfg.traffic_light_binding_root) if cfg.traffic_light_binding_root is not None else "disabled"
    map_admission_source = str(cfg.training_mapping_table) if cfg.training_mapping_table is not None else "default_parser_scope"

    global_counts = CoverageCounts(records=len(records))
    city_counts: dict[str, CoverageCounts] = defaultdict(CoverageCounts)
    record_counts: dict[str, CoverageCounts] = {}

    for record in records:
        key = f"{record.city}/{record.record_name}"
        record_counter = CoverageCounts(records=1)
        city_counts[record.city].records += 1
        windows = windows_for_record(
            record,
            past_len=cfg.past_len,
            future_len=cfg.future_len,
            stride=cfg.stride,
            max_scenarios=args.max_scenarios_per_record,
            admission_rules=admission_rules,
            traffic_light_binding_root=cfg.traffic_light_binding_root,
        )
        for window in windows:
            dynamic_map_states = dynamic_map_states_for_window(
                window.traffic_light,
                window.traffic_light_bindings,
                window.timestamps_ms,
            )
            has_signal_objects, has_known_any, informative_center = _count_dynamic_map_states(
                dynamic_map_states,
                current_time_index=window.past_len - 1,
            )
            target_count = len(window.target_ids)
            _accumulate(record_counter, target_count, has_signal_objects, has_known_any, informative_center)
            _accumulate(city_counts[record.city], target_count, has_signal_objects, has_known_any, informative_center)
            _accumulate(global_counts, target_count, has_signal_objects, has_known_any, informative_center)
        record_counts[key] = record_counter

    rows = [_summary_row("global", "ALL", global_counts)]
    rows.extend(_summary_row("city", city, counts) for city, counts in sorted(city_counts.items()))
    rows.extend(_summary_row("record", name, counts) for name, counts in sorted(record_counts.items()))

    global_row = rows[0]
    print(
        json.dumps(
            {
                "scope": global_row["scope"],
                "name": global_row["name"],
                "records": global_row["records"],
                "scenarios": global_row["scenarios"],
                "target_samples": global_row["target_samples"],
                "signal_binding_source": signal_binding_source,
                "map_admission_source": map_admission_source,
                "scenario_signal_object_coverage": global_row["scenario_signal_object_coverage"],
                "scenario_known_state_any_frame_coverage": global_row["scenario_known_state_any_frame_coverage"],
                "scenario_informative_center_coverage": global_row["scenario_informative_center_coverage"],
                "sample_signal_object_coverage": global_row["sample_signal_object_coverage"],
                "sample_known_state_any_frame_coverage": global_row["sample_known_state_any_frame_coverage"],
                "sample_informative_center_coverage": global_row["sample_informative_center_coverage"],
            },
            indent=2,
        )
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "signal_binding_source": signal_binding_source,
                    "map_admission_source": map_admission_source,
                    "rows": rows,
                },
                f,
                indent=2,
            )

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
