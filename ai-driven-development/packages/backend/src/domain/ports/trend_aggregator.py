"""Trend aggregator port -- anonymized community trend data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class GenreTrendItem:
    genre: str
    play_count: int
    change_pct: float  # % change vs previous period


@dataclass(frozen=True)
class GenreTrendReport:
    region: str
    days: int
    trends: list[GenreTrendItem]


@dataclass(frozen=True)
class BpmBucket:
    bpm_min: float
    bpm_max: float
    count: int


@dataclass(frozen=True)
class BpmDistribution:
    genre: str
    days: int
    buckets: list[BpmBucket]
    mean_bpm: float
    median_bpm: float


@dataclass(frozen=True)
class AnonymizedTrack:
    fingerprint: str
    genre: str
    bpm: float
    key: str
    play_count: int


@dataclass(frozen=True)
class TrendingTransition:
    track_a_fingerprint: str
    track_b_fingerprint: str
    frequency: int
    genre: str


class TrendAggregator(ABC):
    @abstractmethod
    async def genre_trends(self, region: str | None, days: int) -> GenreTrendReport: ...

    @abstractmethod
    async def bpm_distribution(self, genre: str | None, days: int) -> BpmDistribution: ...

    @abstractmethod
    async def top_tracks(
        self, genre: str | None, days: int, limit: int
    ) -> list[AnonymizedTrack]: ...

    @abstractmethod
    async def trending_transitions(
        self, genre: str | None, days: int, limit: int
    ) -> list[TrendingTransition]: ...
