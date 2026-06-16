from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sind_converter.data.discovery import RecordDescription


@dataclass(frozen=True)
class LoadedRecord:
    description: RecordDescription
    vehicle_tracks: pd.DataFrame
    vehicle_meta: pd.DataFrame
    pedestrian_tracks: pd.DataFrame
    pedestrian_meta: pd.DataFrame
    traffic_light: pd.DataFrame | None
    recording_meta: pd.Series | None


def _infer_meta_from_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for track_id, group in tracks.groupby("track_id", sort=True):
        first = group.iloc[0]
        rows.append(
            {
                "trackId": str(track_id),
                "class": str(first.get("agent_type", "unknown")).strip().lower(),
                "initialFrame": int(group["frame_id"].min()),
                "finalFrame": int(group["frame_id"].max()),
                "numFrames": int(group["frame_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _read_optional_csv(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def _normalize_track_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["track_id"] = df["track_id"].map(str)
    return df


def load_record(description: RecordDescription) -> LoadedRecord:
    vehicle_tracks = _normalize_track_ids(pd.read_csv(description.vehicle_tracks_path))
    pedestrian_tracks = _normalize_track_ids(pd.read_csv(description.pedestrian_tracks_path))
    vehicle_meta = _read_optional_csv(description.vehicle_meta_path)
    pedestrian_meta = _read_optional_csv(description.pedestrian_meta_path)
    traffic_light = _read_optional_csv(description.traffic_light_path)
    recording_meta_df = _read_optional_csv(description.recording_meta_path)

    if vehicle_meta is None:
        vehicle_meta = _infer_meta_from_tracks(vehicle_tracks)
    else:
        vehicle_meta = vehicle_meta.copy()
        id_col = "trackId" if "trackId" in vehicle_meta.columns else "track_id"
        class_col = "class" if "class" in vehicle_meta.columns else "agent_type"
        vehicle_meta["trackId"] = vehicle_meta[id_col].map(str)
        vehicle_meta["class"] = vehicle_meta[class_col].map(lambda value: str(value).strip().lower())

    if pedestrian_meta is None:
        pedestrian_meta = _infer_meta_from_tracks(pedestrian_tracks)

    recording_meta = None if recording_meta_df is None or recording_meta_df.empty else recording_meta_df.iloc[0]
    return LoadedRecord(description, vehicle_tracks, vehicle_meta, pedestrian_tracks, pedestrian_meta, traffic_light, recording_meta)
