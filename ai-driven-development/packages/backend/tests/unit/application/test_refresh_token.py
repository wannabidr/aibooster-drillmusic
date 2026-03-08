"""Tests for RefreshAccessToken use case."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from src.application.dto.auth_request import RefreshRequest
from src.application.use_cases.refresh_access_token import RefreshAccessToken
from src.domain.entities.session import Session


@pytest.fixture
def refresh_use_case(
    mock_session_repo: AsyncMock,
    mock_token_service: AsyncMock,
) -> RefreshAccessToken:
    return RefreshAccessToken(
        session_repo=mock_session_repo,
        token_service=mock_token_service,
    )


class TestRefreshAccessToken:
    @pytest.mark.asyncio
    async def test_valid_refresh(
        self,
        refresh_use_case: RefreshAccessToken,
        mock_session_repo: AsyncMock,
        sample_session: Session,
    ) -> None:
        mock_session_repo.find_by_refresh_hash.return_value = sample_session

        result = await refresh_use_case.execute(RefreshRequest(refresh_token="valid-refresh-token"))

        assert result.access_token == "new-access-token"
        assert mock_session_repo.save.call_count == 2  # revoke old + save new

    @pytest.mark.asyncio
    async def test_invalid_token_raises(
        self,
        refresh_use_case: RefreshAccessToken,
        mock_session_repo: AsyncMock,
    ) -> None:
        mock_session_repo.find_by_refresh_hash.return_value = None

        with pytest.raises(PermissionError, match="Invalid refresh token"):
            await refresh_use_case.execute(RefreshRequest(refresh_token="invalid-token"))

    @pytest.mark.asyncio
    async def test_revoked_session_raises(
        self,
        refresh_use_case: RefreshAccessToken,
        mock_session_repo: AsyncMock,
        sample_session: Session,
    ) -> None:
        revoked = sample_session.revoke()
        mock_session_repo.find_by_refresh_hash.return_value = revoked

        with pytest.raises(PermissionError, match="revoked"):
            await refresh_use_case.execute(RefreshRequest(refresh_token="revoked-token"))

    @pytest.mark.asyncio
    async def test_expired_session_raises(
        self,
        refresh_use_case: RefreshAccessToken,
        mock_session_repo: AsyncMock,
        expired_session: Session,
    ) -> None:
        mock_session_repo.find_by_refresh_hash.return_value = expired_session

        with pytest.raises(PermissionError, match="expired"):
            await refresh_use_case.execute(RefreshRequest(refresh_token="expired-token"))
