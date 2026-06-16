from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RecordDescription:
    city: str
    record_name: str
    record_dir: Path
    vehicle_tracks_path: Path
    pedestrian_tracks_path: Path
    traffic_light_path: Path | None
    vehicle_meta_path: Path | None
    pedestrian_meta_path: Path | None
    recording_meta_path: Path | None
    map_path: Path


MAP_CANDIDATES = {
    "Tianjin": [("fallback", "Tianjin/map_relink_law_save.osm")],
    "Changchun": [("data", "Changchun/Changchun_Pudong.osm"), ("fallback", "Changchun/Changchun_Pudong.osm")],
    "Chongqing": [("data", "Chongqing/NR_ll2.osm"), ("fallback", "Chongqing/NR_ll2.osm")],
    "Xi_an": [("data", "Xi_an/Xi_an_Shanglin.osm"), ("fallback", "Xi'an/Xi'an_Shanglin.osm")],
}


def normalize_city_name(name: str) -> str:
    if name in {"Xi'an", "Xian", "Xi_an"}:
        return "Xi_an"
    return name


def resolve_map_path(city: str, data_root: Path, map_fallback_root: Path) -> Path:
    normalized_city = normalize_city_name(city)
    for root_kind, rel_path in MAP_CANDIDATES.get(normalized_city, []):
        root = data_root if root_kind == "data" else map_fallback_root
        candidate = root / rel_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No OSM map found for city={city!r}")


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _find_traffic_light(record_dir: Path) -> Path | None:
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


def discover_records(data_root: Path, map_fallback_root: Path, cities: list[str] | None = None) -> list[RecordDescription]:
    data_root = Path(data_root)
    map_fallback_root = Path(map_fallback_root)
    city_dirs = [data_root / city for city in cities] if cities else sorted([p for p in data_root.iterdir() if p.is_dir()])
    records: list[RecordDescription] = []
    for city_dir in city_dirs:
        if not city_dir.is_dir():
            continue
        city = normalize_city_name(city_dir.name)
        map_path = resolve_map_path(city, data_root, map_fallback_root)
        for record_dir in sorted([p for p in city_dir.iterdir() if p.is_dir()]):
            vehicle_tracks = record_dir / "Veh_smoothed_tracks.csv"
            pedestrian_tracks = record_dir / "Ped_smoothed_tracks.csv"
            if not vehicle_tracks.exists() or not pedestrian_tracks.exists():
                continue
            records.append(
                RecordDescription(
                    city=city,
                    record_name=record_dir.name,
                    record_dir=record_dir,
                    vehicle_tracks_path=vehicle_tracks,
                    pedestrian_tracks_path=pedestrian_tracks,
                    traffic_light_path=_find_traffic_light(record_dir),
                    vehicle_meta_path=_first_existing([record_dir / "Veh_tracks_meta.csv"]),
                    pedestrian_meta_path=_first_existing([record_dir / "Ped_tracks_meta.csv"]),
                    recording_meta_path=_first_existing([record_dir / "recording_metas.csv", record_dir / "recoding_metas.csv"]),
                    map_path=map_path,
                )
            )
    return records
