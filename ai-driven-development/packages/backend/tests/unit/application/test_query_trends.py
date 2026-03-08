"""Tests for QueryTrends use case."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.application.use_cases.query_trends import QueryTrends
from src.domain.ports.trend_aggregator import (
    AnonymizedTrack,
    BpmBucket,
    BpmDistribution,
    GenreTrendItem,
    GenreTrendReport,
    TrendingTransition,
)


@pytest.fixture
def mock_aggregator() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def use_case(mock_aggregator: AsyncMock) -> QueryTrends:
    return QueryTrends(trend_aggregator=mock_aggregator)


class TestGenreTrends:
    async def test_returns_trend_dto(
        self, use_case: QueryTrends, mock_aggregator: AsyncMock
    ) -> None:
        mock_aggregator.genre_trends.return_value = GenreTrendReport(
            region="global",
            days=30,
            trends=[
                GenreTrendItem(genre="techno", play_count=1000, change_pct=15.2),
                GenreTrendItem(genre="house", play_count=800, change_pct=-3.1),
            ],
        )

        result = await use_case.genre_trends(None, 30)

        assert result.region == "global"
        assert result.days == 30
        assert len(result.trends) == 2
        assert result.trends[0].genre == "techno"
        assert result.trends[0].play_count == 1000
        assert result.trends[0].change_pct == 15.2

    async def test_passes_region_filter(
        self, use_case: QueryTrends, mock_aggregator: AsyncMock
    ) -> None:
        mock_aggregator.genre_trends.return_value = GenreTrendReport(
            region="eu", days=7, trends=[]
        )

        await use_case.genre_trends("eu", 7)
        mock_aggregator.genre_trends.assert_called_once_with("eu", 7)


class TestBpmDistribution:
    async def test_returns_distribution_dto(
        self, use_case: QueryTrends, mock_aggregator: AsyncMock
    ) -> None:
        mock_aggregator.bpm_distribution.return_value = BpmDistribution(
            genre="techno",
            days=30,
            buckets=[
                BpmBucket(bpm_min=130.0, bpm_max=135.0, count=500),
                BpmBucket(bpm_min=135.0, bpm_max=140.0, count=300),
            ],
            mean_bpm=133.5,
            median_bpm=132.0,
        )

        result = await use_case.bpm_distribution("techno", 30)

        assert result.genre == "techno"
        assert len(result.buckets) == 2
        assert result.mean_bpm == 133.5
        assert result.median_bpm == 132.0

    async def test_empty_distribution(
        self, use_case: QueryTrends, mock_aggregator: AsyncMock
    ) -> None:
        mock_aggregator.bpm_distribution.return_value = BpmDistribution(
            genre="all", days=30, buckets=[], mean_bpm=0.0, median_bpm=0.0
        )

        result = await use_case.bpm_distribution(None, 30)
        assert result.buckets == []
        assert result.mean_bpm == 0.0


class TestTopTracks:
    async def test_returns_anonymized_tracks(
        self, use_case: QueryTrends, mock_aggregator: AsyncMock
    ) -> None:
        mock_aggregator.top_tracks.return_value = [
            AnonymizedTrack(
                fingerprint="fp_abc", genre="techno", bpm=138.0, key="Am", play_count=200
            ),
        ]

        result = await use_case.top_tracks("techno", 30, 10)

        assert len(result) == 1
        assert result[0].fingerprint == "fp_abc"
        assert result[0].play_count == 200

    async def test_passes_limit(
        self, use_case: QueryTrends, mock_aggregator: AsyncMock
    ) -> None:
        mock_aggregator.top_tracks.return_value = []
        await use_case.top_tracks(None, 7, 50)
        mock_aggregator.top_tracks.assert_called_once_with(None, 7, 50)


class TestTrendingTransitions:
    async def test_returns_transitions(
        self, use_case: QueryTrends, mock_aggregator: AsyncMock
    ) -> None:
        mock_aggregator.trending_transitions.return_value = [
            TrendingTransition(
                track_a_fingerprint="fp_a",
                track_b_fingerprint="fp_b",
                frequency=42,
                genre="house",
            ),
        ]

        result = await use_case.trending_transitions("house", 30, 20)

        assert len(result) == 1
        assert result[0].track_a_fingerprint == "fp_a"
        assert result[0].frequency == 42
