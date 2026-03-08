"""Tests for GetCurrentUser use case."""

from __future__ import annotations

from datetime import UTC
from unittest.mock import AsyncMock

import pytest
from src.application.use_cases.get_current_user import GetCurrentUser
from src.domain.entities.user import User


@pytest.fixture
def get_user_use_case(
    mock_token_service: AsyncMock,
    mock_user_repo: AsyncMock,
) -> GetCurrentUser:
    return GetCurrentUser(
        token_service=mock_token_service,
        user_repo=mock_user_repo,
    )


class TestGetCurrentUser:
    @pytest.mark.asyncio
    async def test_valid_token_returns_user(
        self,
        get_user_use_case: GetCurrentUser,
        mock_user_repo: AsyncMock,
        sample_user: User,
    ) -> None:
        mock_user_repo.find_by_id.return_value = sample_user

        result = await get_user_use_case.execute("valid-access-token")

        assert result.email.value == "dj@example.com"

    @pytest.mark.asyncio
    async def test_user_not_found_raises(
        self,
        get_user_use_case: GetCurrentUser,
        mock_user_repo: AsyncMock,
    ) -> None:
        mock_user_repo.find_by_id.return_value = None

        with pytest.raises(PermissionError, match="User not found"):
            await get_user_use_case.execute("valid-access-token")

    @pytest.mark.asyncio
    async def test_deleted_user_raises(
        self,
        get_user_use_case: GetCurrentUser,
        mock_user_repo: AsyncMock,
        sample_user: User,
    ) -> None:
        from datetime import datetime

        deleted = sample_user.mark_deleted(datetime.now(UTC))
        mock_user_repo.find_by_id.return_value = deleted

        with pytest.raises(PermissionError, match="User not found"):
            await get_user_use_case.execute("valid-access-token")

    @pytest.mark.asyncio
    async def test_execute_returns_user_entity(
        self,
        get_user_use_case: GetCurrentUser,
        mock_user_repo: AsyncMock,
        sample_user: User,
    ) -> None:
        mock_user_repo.find_by_id.return_value = sample_user

        result = await get_user_use_case.execute("valid-access-token")

        assert isinstance(result, User)
        assert result.email.value == "dj@example.com"
