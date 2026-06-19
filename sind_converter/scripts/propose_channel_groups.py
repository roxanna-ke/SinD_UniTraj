from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate candidate traffic-light channel_groups JSON files from audit CSVs.")
    parser.add_argument(
        "--mapping-csv",
        type=Path,
        default=Path("output/lights/traffic_light_channel_mapping.csv"),
        help="CSV produced by infer_sind_traffic_light_mapping.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sind_converter/lights/config/channel_groups"),
        help="Directory to write per-city JSON files",
    )
    parser.add_argument("--min-support", type=int, default=2, help="Minimum aggregated count to keep a movement candidate")
    parser.add_argument(
        "--per-stopline-ratio",
        type=float,
        default=0.5,
        help="Within each channel+stopline, keep movements whose support is at least this ratio of the strongest movement",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(args.mapping_csv.open(newline="", encoding="utf-8")))
    if not rows:
        raise SystemExit(f"No rows found in {args.mapping_csv}")

    grouped: dict[tuple[str, str], dict[tuple[str, str], int]] = defaultdict(lambda: defaultdict(int))
    channel_type: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["city"], row["traffic_light_channel"])
        movement_key = (str(row["stopline_id"]), str(row["movement"]).strip().lower())
        grouped[key][movement_key] += 1
        channel_type[key] = row.get("channel_type", "vehicle")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_city: dict[str, list[dict]] = defaultdict(list)
    for (city, channel), counts in sorted(grouped.items()):
        if channel_type.get((city, channel), "vehicle") != "vehicle":
            continue
        by_stopline: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for (stopline_id, movement), support in counts.items():
            by_stopline[stopline_id].append((movement, support))

        selected_movements = []
        total_support = 0
        for stopline_id, movement_rows in sorted(by_stopline.items()):
            best_support = max(support for _, support in movement_rows)
            threshold = max(args.min_support, int(math.ceil(best_support * args.per_stopline_ratio)))
            for movement, support in sorted(movement_rows, key=lambda item: (-item[1], item[0])):
                if support < threshold:
                    continue
                total_support += support
                selected_movements.append(
                    {
                        "stopline_id": stopline_id,
                        "movement": movement,
                        "support": support,
                    }
                )
        if not selected_movements:
            continue

        by_city[city].append(
            {
                "group_id": _group_id(channel),
                "traffic_light_channels": [channel],
                "movements": selected_movements,
                "confidence": _confidence(total_support),
                "source": "candidate_from_traffic_light_channel_mapping_csv",
            }
        )

    for city, groups in sorted(by_city.items()):
        payload = {
            "city": city,
            "source_csv": str(args.mapping_csv),
            "channel_groups": groups,
        }
        output_path = args.output_dir / f"{city}.json"
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
        print(f"[ok] wrote {output_path} with {len(groups)} groups")


def _group_id(channel: str) -> str:
    lowered = "".join(ch.lower() if ch.isalnum() else "_" for ch in channel)
    collapsed = "_".join(part for part in lowered.split("_") if part)
    return f"group_{collapsed}"


def _confidence(total_support: int) -> str:
    if total_support >= 30:
        return "high"
    if total_support >= 10:
        return "medium"
    return "low"


if __name__ == "__main__":
    main()
