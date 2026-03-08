"""BPM (Beats Per Minute) value object."""

from __future__ import annotations

from dataclasses import dataclass

MIN_BPM = 20.0
MAX_BPM = 300.0


@dataclass(frozen=True)
class BPMValue:
    value: float

    def __post_init__(self) -> None:
        if not MIN_BPM <= self.value <= MAX_BPM:
            raise ValueError(f"BPM must be between {MIN_BPM} and {MAX_BPM}, got {self.value}")

    def half_time(self) -> BPMValue:
        return BPMValue(self.value / 2)

    def double_time(self) -> BPMValue:
        return BPMValue(self.value * 2)
