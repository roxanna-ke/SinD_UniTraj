#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNITRAJ_ROOT = PROJECT_ROOT / "UniTraj"
SCENARIONET_ROOT = PROJECT_ROOT / "scenarionet"

for path in (PROJECT_ROOT, UNITRAJ_ROOT, SCENARIONET_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scenarionet.common_utils import read_dataset_summary, read_scenario
from unitraj.datasets.types import traffic_light_state_to_int


LIGHT_SLICE_START = 9
LIGHT_FEATURE_DIM = len(traffic_light_state_to_int)


def _decode_scalar(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_scalar(value.item())
        if value.size == 1:
            return _decode_scalar(value.reshape(-1)[0])
    return str(value)


def _scenario_light_summary(scenario: dict, fallback_current_time_index: int) -> tuple[list[dict], bool]:
    current_time_index = int(scenario.get("metadata", {}).get("current_time_index", fallback_current_time_index))
    current_time_index = max(current_time_index, 0)

    lane_rows: list[dict] = []
    informative = False
    for signal_id, signal in scenario.get("dynamic_map_states", {}).items():
        lane_id = str(signal.get("lane", ""))
        states = list(signal.get("state", {}).get("object_state", []))
        if not lane_id or not states:
            continue
        state_idx = min(current_time_index, len(states) - 1)
        raw_state = states[state_idx]
        state_int = int(traffic_light_state_to_int.get(raw_state, 0))
        informative = informative or state_int > 0
        lane_rows.append(
            {
                "signal_id": str(signal_id),
                "lane_id": lane_id,
                "state_raw": str(raw_state),
                "state_int": state_int,
            }
        )
    return lane_rows, informative


def _load_informative_scenarios(split_dir: Path, sample_count: int, past_len: int) -> list[dict]:
    _, scenario_files, mapping = read_dataset_summary(str(split_dir))
    informative_rows: list[dict] = []
    for scenario_file in scenario_files:
        scenario = read_scenario(str(split_dir), mapping, scenario_file)
        lane_rows, informative = _scenario_light_summary(scenario, fallback_current_time_index=past_len - 1)
        if not informative:
            continue
        informative_rows.append(
            {
                "scenario_id": str(scenario.get("metadata", {}).get("scenario_id", scenario_file)),
                "scenario_file": scenario_file,
                "signal_count": len(lane_rows),
                "active_signal_count": sum(1 for row in lane_rows if row["state_int"] > 0),
                "signals": lane_rows,
            }
        )
        if len(informative_rows) >= sample_count:
            break
    return informative_rows


def _cache_has_signal_features(map_polylines: np.ndarray, map_polylines_mask: np.ndarray) -> bool:
    if map_polylines.ndim != 3:
        raise ValueError(f"Expected map_polylines to have 3 dims, got shape={map_polylines.shape}")
    if map_polylines.shape[-1] < LIGHT_SLICE_START + LIGHT_FEATURE_DIM:
        raise ValueError(
            "map_polylines feature dim is too small to contain lane-control one-hot features: "
            f"shape={map_polylines.shape}"
        )

    light_slice = map_polylines[..., LIGHT_SLICE_START:LIGHT_SLICE_START + LIGHT_FEATURE_DIM]
    valid_mask = map_polylines_mask.astype(bool)
    if light_slice.shape[:2] != valid_mask.shape:
        raise ValueError(
            "map_polylines and map_polylines_mask shape mismatch: "
            f"{map_polylines.shape} vs {map_polylines_mask.shape}"
        )

    if not np.any(valid_mask):
        return False

    light_class = np.argmax(light_slice, axis=-1)
    light_mass = np.sum(light_slice, axis=-1)
    return bool(np.any(valid_mask & (light_mass > 0.5) & (light_class > 0)))


def _cache_has_informative_light_tokens(group: h5py.Group) -> bool:
    required = {"light_token_features", "light_token_mask", "light_token_valid_mask"}
    if not required.issubset(group.keys()):
        return False

    light_token_features = group["light_token_features"][()]
    light_token_mask = group["light_token_mask"][()].astype(bool)
    light_token_valid_mask = group["light_token_valid_mask"][()].astype(bool)
    if light_token_features.ndim != 3:
        raise ValueError(
            f"Expected light_token_features to have 3 dims for one sample, got shape={light_token_features.shape}"
        )

    light_state_slice = light_token_features[..., :LIGHT_FEATURE_DIM]
    light_state_class = np.argmax(light_state_slice, axis=-1)
    light_state_mass = np.sum(light_state_slice, axis=-1)
    token_mask = light_token_mask[..., None] & light_token_valid_mask
    return bool(np.any(token_mask & (light_state_mass > 0.5) & (light_state_class > 0)))


def _scan_cache(cache_dir: Path) -> dict[str, list[dict]]:
    file_list_path = cache_dir / "file_list.pkl"
    if not file_list_path.exists():
        raise FileNotFoundError(f"Missing cache index: {file_list_path}")

    with file_list_path.open("rb") as f:
        file_list = pickle.load(f)

    scenario_hits: dict[str, list[dict]] = defaultdict(list)
    for group_name, file_info in file_list.items():
        h5_path = Path(file_info["h5_path"])
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing HDF5 cache shard: {h5_path}")

        with h5py.File(h5_path, "r") as h5_file:
            if group_name not in h5_file:
                raise KeyError(f"Cache group {group_name} not found in {h5_path}")
            group = h5_file[group_name]
            scenario_id = _decode_scalar(group["scenario_id"][()])
            map_polylines = group["map_polylines"][()]
            map_polylines_mask = group["map_polylines_mask"][()]
            has_signal_features = _cache_has_signal_features(map_polylines, map_polylines_mask)
            has_light_tokens = _cache_has_informative_light_tokens(group)
            scenario_hits[scenario_id].append(
                {
                    "group_name": str(group_name),
                    "h5_path": str(h5_path),
                    "has_signal_features": has_signal_features,
                    "has_light_tokens": has_light_tokens,
                }
            )
    return scenario_hits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether informative traffic-light states survive from ScenarioNet into final UniTraj caches."
    )
    parser.add_argument("--split-dir", type=Path, required=True, help="ScenarioNet split directory, e.g. .../train/sind")
    parser.add_argument("--cache-dir", type=Path, required=True, help="UniTraj cache directory, e.g. .../MTR/sind/train")
    parser.add_argument("--sample-count", type=int, default=8, help="Number of informative scenarios to verify.")
    parser.add_argument("--past-len", type=int, default=21, help="Fallback current-time index source when metadata is missing.")
    parser.add_argument(
        "--require-light-tokens",
        action="store_true",
        help="Fail unless informative history-based light_token_* features are present in the final cache.",
    )
    args = parser.parse_args()

    informative_scenarios = _load_informative_scenarios(args.split_dir, args.sample_count, args.past_len)
    if not informative_scenarios:
        raise SystemExit(
            f"No informative traffic-light scenarios found in {args.split_dir}. "
            "The split exists, but none of the sampled scenarios had a center-frame light state > 0."
        )

    scenario_hits = _scan_cache(args.cache_dir)
    missing_map_tokens: list[dict] = []
    missing_light_tokens: list[dict] = []

    print(f"[info] split_dir={args.split_dir}")
    print(f"[info] cache_dir={args.cache_dir}")
    print(f"[info] informative_scenarios_checked={len(informative_scenarios)}")

    for row in informative_scenarios:
        scenario_id = row["scenario_id"]
        cache_rows = scenario_hits.get(scenario_id, [])
        matched = any(cache_row["has_signal_features"] for cache_row in cache_rows)
        matched_light_tokens = any(cache_row["has_light_tokens"] for cache_row in cache_rows)
        active_states = sorted({signal["state_int"] for signal in row["signals"] if signal["state_int"] > 0})
        print(
            "[check] "
            f"scenario_id={scenario_id} active_signal_count={row['active_signal_count']} "
            f"active_state_ids={active_states} cache_samples={len(cache_rows)} "
            f"matched_map_tokens={matched} matched_light_tokens={matched_light_tokens}"
        )
        if not matched:
            missing_map_tokens.append(row)
        if args.require_light_tokens and not matched_light_tokens:
            missing_light_tokens.append(row)

    total_cache_samples = sum(len(rows) for rows in scenario_hits.values())
    positive_cache_samples = sum(
        1 for rows in scenario_hits.values() for row in rows if row["has_signal_features"]
    )
    positive_light_token_samples = sum(
        1 for rows in scenario_hits.values() for row in rows if row["has_light_tokens"]
    )
    print(
        "[summary] "
        f"cache_scenarios={len(scenario_hits)} cache_samples={total_cache_samples} "
        f"cache_samples_with_signal_features={positive_cache_samples} "
        f"cache_samples_with_light_tokens={positive_light_token_samples}"
    )

    if missing_map_tokens or missing_light_tokens:
        if args.require_light_tokens and missing_light_tokens:
            raise SystemExit(
                "Signal information did not survive into the final UniTraj cache with the expected schema for "
                f"{len(missing_light_tokens)} checked scenario(s). Example: {missing_light_tokens[0]['scenario_id']} reason=light_tokens"
            )
        if missing_map_tokens:
            raise SystemExit(
                "Signal information did not survive into the final UniTraj cache for "
                f"{len(missing_map_tokens)} checked scenario(s). Example: {missing_map_tokens[0]['scenario_id']}"
            )

    if args.require_light_tokens:
        print("[done] informative traffic-light states are present in both map-token and history light-token cache fields")
    else:
        print("[done] informative traffic-light states are present in the final UniTraj cache")


if __name__ == "__main__":
    main()
