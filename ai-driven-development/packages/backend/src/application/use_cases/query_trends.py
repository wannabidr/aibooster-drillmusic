"""Query trend analytics use case -- B2B API."""

from __future__ import annotations

from src.application.dto.trend_dto import (
    AnonymizedTrackDTO,
    BpmBucketDTO,
    BpmDistributionResponseDTO,
    GenreTrendItemDTO,
    GenreTrendResponseDTO,
    TrendingTransitionDTO,
)
from src.domain.ports.trend_aggregator import TrendAggregator


class QueryTrends:
    def __init__(self, trend_aggregator: TrendAggregator) -> None:
        self._aggregator = trend_aggregator

    async def genre_trends(
        self, region: str | None, days: int
    ) -> GenreTrendResponseDTO:
        report = await self._aggregator.genre_trends(region, days)
        return GenreTrendResponseDTO(
            region=report.region,
            days=report.days,
            trends=[
                GenreTrendItemDTO(
                    genre=t.genre,
                    play_count=t.play_count,
                    change_pct=t.change_pct,
                )
                for t in report.trends
            ],
        )

    async def bpm_distribution(
        self, genre: str | None, days: int
    ) -> BpmDistributionResponseDTO:
        dist = await self._aggregator.bpm_distribution(genre, days)
        return BpmDistributionResponseDTO(
            genre=dist.genre,
            days=dist.days,
            buckets=[
                BpmBucketDTO(
                    bpm_min=b.bpm_min,
                    bpm_max=b.bpm_max,
                    count=b.count,
                )
                for b in dist.buckets
            ],
            mean_bpm=dist.mean_bpm,
            median_bpm=dist.median_bpm,
        )

    async def top_tracks(
        self, genre: str | None, days: int, limit: int
    ) -> list[AnonymizedTrackDTO]:
        tracks = await self._aggregator.top_tracks(genre, days, limit)
        return [
            AnonymizedTrackDTO(
                fingerprint=t.fingerprint,
                genre=t.genre,
                bpm=t.bpm,
                key=t.key,
                play_count=t.play_count,
            )
            for t in tracks
        ]

    async def trending_transitions(
        self, genre: str | None, days: int, limit: int
    ) -> list[TrendingTransitionDTO]:
        transitions = await self._aggregator.trending_transitions(genre, days, limit)
        return [
            TrendingTransitionDTO(
                track_a_fingerprint=t.track_a_fingerprint,
                track_b_fingerprint=t.track_b_fingerprint,
                frequency=t.frequency,
                genre=t.genre,
            )
            for t in transitions
        ]
