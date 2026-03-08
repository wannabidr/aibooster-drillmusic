"""DTOs for B2B trend analytics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrendRequest:
    genre: str | None = None
    region: str | None = None
    days: int = 30
    limit: int = 20


@dataclass(frozen=True)
class GenreTrendItemDTO:
    genre: str
    play_count: int
    change_pct: float


@dataclass(frozen=True)
class GenreTrendResponseDTO:
    region: str | None
    days: int
    trends: list[GenreTrendItemDTO]


@dataclass(frozen=True)
class BpmBucketDTO:
    bpm_min: float
    bpm_max: float
    count: int


@dataclass(frozen=True)
class BpmDistributionResponseDTO:
    genre: str | None
    days: int
    buckets: list[BpmBucketDTO]
    mean_bpm: float
    median_bpm: float


@dataclass(frozen=True)
class AnonymizedTrackDTO:
    fingerprint: str
    genre: str
    bpm: float
    key: str
    play_count: int


@dataclass(frozen=True)
class TrendingTransitionDTO:
    track_a_fingerprint: str
    track_b_fingerprint: str
    frequency: int
    genre: str
