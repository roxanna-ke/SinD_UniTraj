from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConverterConfig:
    data_root: Path
    map_fallback_root: Path
    canonical_scenario_root: Path
    split_root: Path
    cache_root: Path
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
