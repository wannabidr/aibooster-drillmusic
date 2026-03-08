"""MixTransition entity -- represents a track-to-track transition in DJ history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MixTransition:
    track_a_hash: str
    track_b_hash: str
    timestamp: datetime
    source: str

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MixTransition):
            return NotImplemented
        return (
            self.track_a_hash == other.track_a_hash
            and self.track_b_hash == other.track_b_hash
            and self.timestamp == other.timestamp
        )

    def __hash__(self) -> int:
        return hash((self.track_a_hash, self.track_b_hash, self.timestamp))
