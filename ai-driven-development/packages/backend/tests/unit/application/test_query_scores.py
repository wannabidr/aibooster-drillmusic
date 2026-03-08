"""Tests for QueryCommunityScores use case."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from src.application.use_cases.query_community_scores import QueryCommunityScores
from src.domain.entities.community_score import CommunityScore


@pytest.fixture
def query_use_case(mock_community_repo: AsyncMock) -> QueryCommunityScores:
    return QueryCommunityScores(community_repo=mock_community_repo)


class TestQueryCommunityScores:
    @pytest.mark.asyncio
    async def test_returns_scores(
        self,
        query_use_case: QueryCommunityScores,
        mock_community_repo: AsyncMock,
    ) -> None:
        mock_community_repo.get_scores_for_track.return_value = [
            CommunityScore(track_a_fingerprint="fp-a", track_b_fingerprint="fp-b", frequency=10),
            CommunityScore(track_a_fingerprint="fp-a", track_b_fingerprint="fp-c", frequency=5),
        ]

        results = await query_use_case.execute("fp-a", limit=20)

        assert len(results) == 2
        assert results[0].frequency == 10
        assert results[1].track_b_fingerprint == "fp-c"

    @pytest.mark.asyncio
    async def test_empty_results(
        self,
        query_use_case: QueryCommunityScores,
        mock_community_repo: AsyncMock,
    ) -> None:
        mock_community_repo.get_scores_for_track.return_value = []

        results = await query_use_case.execute("unknown-fp")

        assert len(results) == 0
