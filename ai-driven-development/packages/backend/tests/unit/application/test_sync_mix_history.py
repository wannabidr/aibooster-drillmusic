"""Tests for SyncMixHistory use case."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from src.application.dto.auth_request import TransitionData
from src.application.use_cases.sync_mix_history import SyncMixHistory
from src.domain.entities.user import User


@pytest.fixture
def sync_use_case(
    mock_community_repo: AsyncMock,
    mock_user_repo: AsyncMock,
) -> SyncMixHistory:
    return SyncMixHistory(
        community_repo=mock_community_repo,
        user_repo=mock_user_repo,
    )


class TestSyncMixHistory:
    @pytest.mark.asyncio
    async def test_sync_transitions(
        self,
        sync_use_case: SyncMixHistory,
        mock_community_repo: AsyncMock,
        sample_user: User,
    ) -> None:
        transitions = [
            TransitionData(track_a_fingerprint="fp-a", track_b_fingerprint="fp-b"),
            TransitionData(track_a_fingerprint="fp-b", track_b_fingerprint="fp-c"),
        ]

        count = await sync_use_case.execute(sample_user, transitions)

        assert count == 2
        mock_community_repo.save_transitions.assert_called_once()
        mock_community_repo.increment_scores.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_transitions(
        self,
        sync_use_case: SyncMixHistory,
        mock_community_repo: AsyncMock,
        sample_user: User,
    ) -> None:
        count = await sync_use_case.execute(sample_user, [])

        assert count == 0
        mock_community_repo.save_transitions.assert_not_called()

    @pytest.mark.asyncio
    async def test_marks_user_as_contributed(
        self,
        sync_use_case: SyncMixHistory,
        mock_user_repo: AsyncMock,
        sample_user: User,
    ) -> None:
        transitions = [
            TransitionData(track_a_fingerprint="fp-a", track_b_fingerprint="fp-b"),
        ]

        await sync_use_case.execute(sample_user, transitions)

        mock_user_repo.save.assert_called_once()
        saved_user = mock_user_repo.save.call_args[0][0]
        assert saved_user.has_contributed

    @pytest.mark.asyncio
    async def test_already_contributed_user_not_updated(
        self,
        sync_use_case: SyncMixHistory,
        mock_user_repo: AsyncMock,
        contributed_user: User,
    ) -> None:
        transitions = [
            TransitionData(track_a_fingerprint="fp-a", track_b_fingerprint="fp-b"),
        ]

        await sync_use_case.execute(contributed_user, transitions)

        mock_user_repo.save.assert_not_called()
