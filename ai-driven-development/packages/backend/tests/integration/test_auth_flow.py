"""Integration tests for auth flow (no real DB, using mocks)."""

from __future__ import annotations

import pytest
from src.application.dto.auth_request import LoginRequest, RefreshRequest
from src.application.use_cases.delete_account import DeleteAccount
from src.application.use_cases.get_current_user import GetCurrentUser
from src.application.use_cases.login_with_oauth import LoginWithOAuth
from src.application.use_cases.logout import Logout
from src.application.use_cases.refresh_access_token import RefreshAccessToken
from src.domain.value_objects.oauth_provider import OAuthProvider

from tests.conftest import (
    MockAuthProvider,
    MockSessionRepository,
    MockTokenService,
    MockUserRepository,
)


@pytest.fixture
def services():
    user_repo = MockUserRepository()
    session_repo = MockSessionRepository()
    token_service = MockTokenService()
    auth_provider = MockAuthProvider(
        email="flow@example.com",
        name="Flow User",
        provider=OAuthProvider.GOOGLE,
        subject="flow-sub",
    )
    return user_repo, session_repo, token_service, auth_provider


@pytest.mark.asyncio
async def test_full_login_refresh_logout_flow(services):
    user_repo, session_repo, token_service, auth_provider = services

    # 1. Login
    login_uc = LoginWithOAuth(
        auth_providers={"google": auth_provider},
        user_repo=user_repo,
        session_repo=session_repo,
        token_service=token_service,
    )
    login_result = await login_uc.execute(
        LoginRequest(provider="google", code="code", redirect_uri="http://localhost")
    )
    assert login_result.user.email == "flow@example.com"
    access_token = login_result.token.access_token
    refresh_token = login_result.token.refresh_token

    # 2. Get current user
    get_user_uc = GetCurrentUser(
        token_service=token_service,
        user_repo=user_repo,
    )
    user = await get_user_uc.execute(access_token)
    assert user.email.value == "flow@example.com"

    # 3. Refresh
    refresh_uc = RefreshAccessToken(
        session_repo=session_repo,
        token_service=token_service,
    )
    refresh_result = await refresh_uc.execute(
        RefreshRequest(refresh_token=refresh_token)
    )
    assert refresh_result.access_token != access_token
    new_refresh = refresh_result.refresh_token

    # 4. Old refresh should fail (rotated)
    with pytest.raises(PermissionError, match="revoked"):
        await refresh_uc.execute(RefreshRequest(refresh_token=refresh_token))

    # 5. Logout with new refresh
    logout_uc = Logout(session_repo=session_repo, token_service=token_service)
    await logout_uc.execute(new_refresh)

    # 6. Refresh after logout should fail
    with pytest.raises(PermissionError, match="revoked"):
        await refresh_uc.execute(RefreshRequest(refresh_token=new_refresh))


@pytest.mark.asyncio
async def test_delete_account_cleanup(services):
    user_repo, session_repo, token_service, auth_provider = services

    # Login
    login_uc = LoginWithOAuth(
        auth_providers={"google": auth_provider},
        user_repo=user_repo,
        session_repo=session_repo,
        token_service=token_service,
    )
    login_result = await login_uc.execute(
        LoginRequest(provider="google", code="code", redirect_uri="http://localhost")
    )

    # Delete account
    delete_uc = DeleteAccount(user_repo=user_repo, session_repo=session_repo)
    from src.domain.value_objects.user_id import UserId

    uid = UserId.from_str(login_result.user.id)
    await delete_uc.execute(uid)

    # User should be gone
    assert await user_repo.find_by_id(uid) is None

    # All sessions should be revoked
    for s in session_repo.sessions.values():
        assert s.revoked is True


@pytest.mark.asyncio
async def test_login_creates_session(services):
    user_repo, session_repo, token_service, auth_provider = services

    login_uc = LoginWithOAuth(
        auth_providers={"google": auth_provider},
        user_repo=user_repo,
        session_repo=session_repo,
        token_service=token_service,
    )
    await login_uc.execute(
        LoginRequest(provider="google", code="code", redirect_uri="http://localhost")
    )

    assert len(session_repo.sessions) == 1
    session = list(session_repo.sessions.values())[0]
    assert session.revoked is False
