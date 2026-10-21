from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def _default_binding_root() -> Path:
    return Path(__file__).resolve().parents[1] / "lights" / "config"


@dataclass(frozen=True)
class ConverterConfig:
    data_root: Path
    map_fallback_root: Path
    canonical_scenario_root: Path
    split_root: Path
    cache_root: Path
    training_mapping_table: Path | None = None
    traffic_light_binding_root: Path | None = field(default_factory=_default_binding_root)
    dataset_name: str = "sind"
    dataset_version: str = "v1"
    past_len: int = 21
    future_len: int = 60
    stride: int = 40

    @property
    def total_length(self) -> int:
        return self.past_len + self.future_len

    @property
    def current_time_index(self) -> int:
        return self.past_len - 1
