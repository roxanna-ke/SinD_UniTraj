from __future__ import annotations

import argparse
from pathlib import Path

from sind_converter.config.defaults import ConverterConfig


def _config_from_args(args: argparse.Namespace) -> ConverterConfig:
    return ConverterConfig(
        data_root=args.data_root,
        map_fallback_root=args.map_fallback_root,
        canonical_scenario_root=args.canonical_scenario_root,
        split_root=args.split_root,
        cache_root=args.cache_root,
        dataset_name=args.dataset_name,
        dataset_version=args.dataset_version,
        past_len=args.past_len,
        future_len=args.future_len,
        stride=args.stride,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SinD modular conversion pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", type=Path, required=True)
    common.add_argument("--map-fallback-root", type=Path, required=True)
    common.add_argument("--canonical-scenario-root", type=Path, required=True)
    common.add_argument("--split-root", type=Path, required=True)
    common.add_argument("--cache-root", type=Path, required=True)
    common.add_argument("--dataset-name", default="sind")
    common.add_argument("--dataset-version", default="v1")
    common.add_argument("--past-len", type=int, default=21)
    common.add_argument("--future-len", type=int, default=60)
    common.add_argument("--stride", type=int, default=40)

    audit = sub.add_parser("audit-maps", parents=[common])
    audit.add_argument("--cities", nargs="*", default=None)
    audit.add_argument("--output-dir", type=Path, required=True)

    convert = sub.add_parser("convert-scenarios", parents=[common])
    convert.add_argument("--cities", nargs="*", default=None)
    convert.add_argument("--max-records", type=int, default=None)
    convert.add_argument("--max-scenarios-per-record", type=int, default=None)

    split = sub.add_parser("make-splits", parents=[common])
    split.add_argument("--canonical-root", type=Path, required=True)
    split.add_argument("--mode", choices=["record-level", "city-holdout"], default="record-level")
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--train-ratio", type=float, default=0.8)
    split.add_argument("--heldout-cities", nargs="*", default=None)
    split.add_argument("--train-cities", nargs="*", default=None)
    split.add_argument("--test-cities", nargs="*", default=None)

    cache = sub.add_parser("build-cache", parents=[common])
    cache.add_argument("--split-dataset-dir", type=Path, required=True)
    cache.add_argument("--unitraj-root", type=Path, required=True)
    cache.add_argument("--unitraj-config", type=Path, required=True)
    cache.add_argument("--method", choices=["autobot", "mtr", "wayformer"], default="autobot")

    args = parser.parse_args()
    cfg = _config_from_args(args)
    if args.command == "audit-maps":
        from sind_converter.data.discovery import discover_records
        from sind_converter.maps.osm import audit_osm_tags

        records = discover_records(cfg.data_root, cfg.map_fallback_root, cities=args.cities)
        audit_osm_tags(sorted({record.map_path for record in records}), args.output_dir)
    elif args.command == "convert-scenarios":
        from sind_converter.scenarios.convert import convert_scenarios

        convert_scenarios(cfg, cities=args.cities, max_records=args.max_records, max_scenarios_per_record=args.max_scenarios_per_record)
    elif args.command == "make-splits":
        from sind_converter.splits.make import make_city_holdout_split, make_record_level_split

        if args.mode == "record-level":
            make_record_level_split(args.canonical_root, cfg.split_root, dataset_name=cfg.dataset_name, train_ratio=args.train_ratio, seed=args.seed)
        else:
            make_city_holdout_split(args.canonical_root, cfg.split_root, args.heldout_cities, args.train_cities, args.test_cities, dataset_name=cfg.dataset_name)
    elif args.command == "build-cache":
        from sind_converter.cache.build import build_unitraj_cache

        build_unitraj_cache(args.split_dataset_dir, args.unitraj_root, args.unitraj_config, method=args.method, cache_root=cfg.cache_root)


if __name__ == "__main__":
    main()
