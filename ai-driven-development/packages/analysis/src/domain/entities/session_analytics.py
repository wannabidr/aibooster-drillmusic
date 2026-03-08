"""SessionAnalytics entity -- aggregated data about a DJ set session."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class TrackPlayEvent:
    track_id: uuid.UUID
    played_at: datetime
    energy: float
    genre: str
    bpm: float
    key: str


@dataclass(frozen=True)
class SessionAnalytics:
    session_id: uuid.UUID
    timestamp: datetime
    tracks_played: list[TrackPlayEvent]
    energy_curve: list[float]
    genre_distribution: dict[str, int]
    bpm_range: tuple[float, float]
    key_transitions: list[tuple[str, str]]
    avg_transition_quality: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SessionAnalytics):
            return NotImplemented
        return self.session_id == other.session_id

    def __hash__(self) -> int:
        return hash(self.session_id)

    @property
    def track_count(self) -> int:
        return len(self.tracks_played)

    @property
    def duration_minutes(self) -> float:
        if len(self.tracks_played) < 2:
            return 0.0
        start = self.tracks_played[0].played_at
        end = self.tracks_played[-1].played_at
        return (end - start).total_seconds() / 60.0


@dataclass(frozen=True)
class AggregateStats:
    total_sessions: int
    total_tracks_played: int
    avg_session_length: float
    top_genres: list[tuple[str, int]]
    avg_bpm: float
    avg_energy: float
    avg_transition_quality: float
    most_common_keys: list[tuple[str, int]]
