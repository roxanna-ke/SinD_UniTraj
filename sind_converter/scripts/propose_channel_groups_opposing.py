"""Generate complete channel_groups JSON files using the two-phase opposing-arm model.

All SinD intersections use a two-phase alternating signal plan: one phase group
controls one pair of opposing arms (e.g. N+S), the other controls the other
pair (e.g. E+W).  Within each phase, straight and left-turn share the green
window.  Right turns are not signal-controlled.

This script assigns every straight and left-turn movement to the correct phase
group, eliminating all "no signal" gaps for signal-controlled movements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Per-city phase configuration
# ---------------------------------------------------------------------------
# Each entry maps a phase-group id to:
#   traffic_light_channels  — SinD CSV column names for this phase
#   stoplines               — stopline IDs whose straight+left movements belong here
#
# Derived from signal cycle analysis and trajectory matching data.

CITY_PHASE_CONFIG: dict[str, list[dict]] = {
    "Changchun": [
        {
            "group_id": "phase_NS",
            "traffic_light_channels": ["Vehicle Traffic light 1"],
            "stoplines": ["-106227", "-106229"],  # S, N
        },
        {
            "group_id": "phase_EW",
            "traffic_light_channels": ["Vehicle Traffic light 2"],
            "stoplines": ["-106228", "-106230"],  # E, W
        },
    ],
    "Chongqing": [
        {
            "group_id": "phase_1",
            "traffic_light_channels": ["Vehicle Traffic light 1", "Vehicle Traffic light 3"],
            "stoplines": ["-104179", "-104199"],  # R2, R4
        },
        {
            "group_id": "phase_2",
            "traffic_light_channels": ["Vehicle Traffic light 2", "Vehicle Traffic light 4"],
            "stoplines": ["-104197", "-104196"],  # R1, R3
        },
    ],
    "Tianjin": [
        {
            "group_id": "phase_E",
            "traffic_light_channels": ["Traffic light 2"],
            "stoplines": ["-124112"],  # E
        },
        {
            "group_id": "phase_N",
            "traffic_light_channels": ["Traffic light 4"],
            "stoplines": ["-124117"],  # N
        },
        {
            "group_id": "phase_W",
            "traffic_light_channels": ["Traffic light 6"],
            "stoplines": ["-124127"],  # W
        },
        {
            "group_id": "phase_S",
            "traffic_light_channels": ["Traffic light 8"],
            "stoplines": ["-124159"],  # S
        },
    ],
    "Xi_an": [
        {
            "group_id": "phase_NS",
            "traffic_light_channels": ["Traffic light 1"],
            "stoplines": ["-103736", "-103745"],  # N, S
        },
        {
            "group_id": "phase_EW",
            "traffic_light_channels": ["Traffic light 2"],
            "stoplines": ["-103740", "-103753"],  # E, W
        },
    ],
}

# Movements that are signal-controlled (right turns are NOT)
SIGNAL_CONTROLLED_MOVEMENTS = {"straight", "left"}


def load_lane_bindings(city: str, binding_root: Path) -> list[dict]:
    """Load lane_bindings for *city*, returning the raw list of binding dicts."""
    for suffix in (".json", ".yaml"):
        path = binding_root / "lane_bindings" / f"{city}{suffix}"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload.get("lane_bindings", [])
    return []


def build_channel_groups(
    city: str,
    binding_root: Path,
) -> dict:
    """Build the complete channel_groups payload for *city*."""
    lane_bindings = load_lane_bindings(city, binding_root)

    # Index: (stopline_id, movement) -> True  for all existing bindings
    existing_pairs: set[tuple[str, str]] = set()
    for binding in lane_bindings:
        existing_pairs.add(
            (str(binding["stopline_id"]), str(binding["movement"]).strip().lower())
        )

    phase_config = CITY_PHASE_CONFIG.get(city)
    if phase_config is None:
        raise ValueError(f"No phase configuration defined for city={city!r}")

    groups: list[dict] = []
    for phase in phase_config:
        stoplines_in_phase = set(phase["stoplines"])
        movements: list[dict] = []

        for stopline_id in phase["stoplines"]:
            for movement in sorted(SIGNAL_CONTROLLED_MOVEMENTS):
                pair = (stopline_id, movement)
                if pair in existing_pairs:
                    movements.append(
                        {"stopline_id": stopline_id, "movement": movement}
                    )

        if not movements:
            continue

        groups.append(
            {
                "group_id": phase["group_id"],
                "traffic_light_channels": phase["traffic_light_channels"],
                "movements": movements,
                "confidence": "high",
                "source": "opposing_arm_two_phase_model",
            }
        )

    return {
        "city": city,
        "channel_groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate channel_groups JSON files using the two-phase opposing-arm model."
    )
    parser.add_argument(
        "--binding-root",
        type=Path,
        default=PROJECT_ROOT / "sind_converter" / "lights" / "config",
        help="Config directory (default: PROJECT_ROOT/sind_converter/lights/config)",
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        default=sorted(CITY_PHASE_CONFIG),
        help="Which cities to generate (default: all 4)",
    )
    args = parser.parse_args()

    output_dir: Path = args.binding_root / "channel_groups"
    output_dir.mkdir(parents=True, exist_ok=True)

    for city in args.cities:
        payload = build_channel_groups(city, args.binding_root)
        output_path = output_dir / f"{city}.json"
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False),
            encoding="utf-8",
        )
        n_groups = len(payload["channel_groups"])
        n_movements = sum(len(g["movements"]) for g in payload["channel_groups"])
        print(f"[ok] {output_path}  — {n_groups} groups, {n_movements} movements")


if __name__ == "__main__":
    main()
