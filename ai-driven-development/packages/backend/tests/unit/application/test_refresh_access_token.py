"""Tests for RefreshAccessToken use case."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from src.application.dto.auth_request import RefreshRequest
from src.application.use_cases.refresh_access_token import RefreshAccessToken

from tests.conftest import MockSessionRepository, MockTokenService, make_session, make_user


@pytest.fixture
def deps():
    session_repo = MockSessionRepository()
    token_service = MockTokenService()
    uc = RefreshAccessToken(session_repo=session_repo, token_service=token_service)
    return uc, session_repo, token_service


@pytest.mark.asyncio
async def test_refresh_valid(deps):
    uc, session_repo, token_service = deps
    user = make_user()
    session = make_session(
        user=user,
        refresh_token_hash="hash:original-refresh",
        expires_at=datetime.now(UTC) + timedelta(days=30),
    )
    await session_repo.save(session)

    result = await uc.execute(RefreshRequest(refresh_token="original-refresh"))
    assert result.access_token
    assert result.refresh_token

    # Old session should be revoked
    old = session_repo.sessions[str(session.id)]
    assert old.revoked is True


@pytest.mark.asyncio
async def test_refresh_invalid_token(deps):
    uc, _, _ = deps
    with pytest.raises(PermissionError, match="Invalid refresh token"):
        await uc.execute(RefreshRequest(refresh_token="nonexistent"))


@pytest.mark.asyncio
async def test_refresh_revoked_session(deps):
    uc, session_repo, _ = deps
    session = make_session(
        refresh_token_hash="hash:revoked-token",
        revoked=True,
    )
    await session_repo.save(session)

    with pytest.raises(PermissionError, match="revoked"):
        await uc.execute(RefreshRequest(refresh_token="revoked-token"))


@pytest.mark.asyncio
async def test_refresh_expired_session(deps):
    uc, session_repo, _ = deps
    session = make_session(
        refresh_token_hash="hash:expired-token",
        expires_at=datetime.now(UTC) - timedelta(hours=1),
    )
    await session_repo.save(session)

    with pytest.raises(PermissionError, match="expired"):
        await uc.execute(RefreshRequest(refresh_token="expired-token"))


@pytest.mark.asyncio
async def test_refresh_creates_new_session(deps):
    uc, session_repo, _ = deps
    session = make_session(
        refresh_token_hash="hash:rotate-me",
        expires_at=datetime.now(UTC) + timedelta(days=30),
    )
    await session_repo.save(session)

    await uc.execute(RefreshRequest(refresh_token="rotate-me"))
    assert len(session_repo.sessions) == 2  # old revoked + new
