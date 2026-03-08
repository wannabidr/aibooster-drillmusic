"""Tests for auth middleware."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from src.domain.value_objects.user_id import UserId
from src.interface.middleware.auth_middleware import get_current_user

from tests.conftest import MockTokenService, MockUserRepository, make_user


@pytest.mark.asyncio
async def test_valid_token_returns_user():
    user_repo = MockUserRepository()
    token_service = MockTokenService()
    user = make_user()
    await user_repo.save(user)

    token = f"access-{user.id}-1"
    result = await get_current_user(
        authorization=f"Bearer {token}",
        token_service=token_service,
        user_repo=user_repo,
    )
    assert result.id == user.id


@pytest.mark.asyncio
async def test_missing_bearer_raises_401():
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(
            authorization="",
            token_service=MockTokenService(),
            user_repo=MockUserRepository(),
        )
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_invalid_token_raises_401():
    """Token that causes verify_access_token to fail should return 401."""
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(
            authorization="Bearer bad-token-no-uuid",
            token_service=MockTokenService(),
            user_repo=MockUserRepository(),
        )
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_user_not_found_raises_401():
    token_service = MockTokenService()
    user_repo = MockUserRepository()
    uid = UserId.generate()
    token = f"access-{uid}-1"

    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(
            authorization=f"Bearer {token}",
            token_service=token_service,
            user_repo=user_repo,
        )
    assert exc_info.value.status_code == 401
