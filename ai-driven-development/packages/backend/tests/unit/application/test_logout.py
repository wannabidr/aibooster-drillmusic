"""Tests for Logout use case."""

from __future__ import annotations

import pytest
from src.application.use_cases.logout import Logout

from tests.conftest import MockSessionRepository, MockTokenService, make_session


@pytest.fixture
def deps():
    session_repo = MockSessionRepository()
    token_service = MockTokenService()
    uc = Logout(session_repo=session_repo, token_service=token_service)
    return uc, session_repo


@pytest.mark.asyncio
async def test_logout_revokes_session(deps):
    uc, session_repo = deps
    session = make_session(refresh_token_hash="hash:logout-token", revoked=False)
    await session_repo.save(session)

    await uc.execute("logout-token")

    updated = session_repo.sessions[str(session.id)]
    assert updated.revoked is True


@pytest.mark.asyncio
async def test_logout_nonexistent_token(deps):
    uc, _ = deps
    # Should not raise
    await uc.execute("nonexistent-token")
