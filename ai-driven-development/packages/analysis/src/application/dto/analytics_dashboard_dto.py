"""Analytics dashboard DTOs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnergyReportDTO:
    session_id: str
    timestamp: str
    energy_curve: list[float]
    peak_energy: float
    valley_energy: float
    avg_energy: float


@dataclass(frozen=True)
class GenreDistributionDTO:
    genres: dict[str, int]
    total_tracks: int


@dataclass(frozen=True)
class MixingPatternsDTO:
    most_common_transitions: list[tuple[str, str, int]]
    preferred_bpm_range: tuple[float, float]
    avg_bpm: float
    key_preferences: list[tuple[str, int]]
    avg_transition_quality: float


@dataclass(frozen=True)
class SessionTimelineItemDTO:
    session_id: str
    timestamp: str
    track_count: int
    duration_minutes: float
    avg_energy: float
    top_genre: str


@dataclass(frozen=True)
class AnalyticsDashboardDTO:
    energy_reports: list[EnergyReportDTO]
    genre_distribution: GenreDistributionDTO
    mixing_patterns: MixingPatternsDTO
    session_timeline: list[SessionTimelineItemDTO]
    total_sessions: int
    total_tracks_played: int
