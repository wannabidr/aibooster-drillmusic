"""Tests for DeleteAccount use case."""

from __future__ import annotations

import pytest
from src.application.use_cases.delete_account import DeleteAccount

from tests.conftest import (
    MockSessionRepository,
    MockUserRepository,
    make_session,
    make_user,
)


@pytest.mark.asyncio
async def test_delete_account_removes_user():
    user_repo = MockUserRepository()
    session_repo = MockSessionRepository()
    user = make_user()
    await user_repo.save(user)

    uc = DeleteAccount(user_repo=user_repo, session_repo=session_repo)
    await uc.execute(user.id)

    assert await user_repo.find_by_id(user.id) is None


@pytest.mark.asyncio
async def test_delete_account_revokes_sessions():
    user_repo = MockUserRepository()
    session_repo = MockSessionRepository()
    user = make_user()
    await user_repo.save(user)

    s1 = make_session(user=user, revoked=False)
    s2 = make_session(user=user, revoked=False)
    await session_repo.save(s1)
    await session_repo.save(s2)

    uc = DeleteAccount(user_repo=user_repo, session_repo=session_repo)
    await uc.execute(user.id)

    for s in session_repo.sessions.values():
        assert s.revoked is True
