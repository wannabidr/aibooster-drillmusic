"""Tests for LoginWithOAuth use case."""

from __future__ import annotations

import pytest
from src.application.dto.auth_request import LoginRequest
from src.application.use_cases.login_with_oauth import LoginWithOAuth
from src.domain.value_objects.oauth_provider import OAuthProvider

from tests.conftest import (
    MockAuthProvider,
    MockSessionRepository,
    MockTokenService,
    MockUserRepository,
    make_user,
)


@pytest.fixture
def use_case():
    return LoginWithOAuth(
        auth_providers={
            "google": MockAuthProvider(
                email="new@gmail.com",
                name="New User",
                provider=OAuthProvider.GOOGLE,
                subject="google-new-sub",
            ),
        },
        user_repo=MockUserRepository(),
        session_repo=MockSessionRepository(),
        token_service=MockTokenService(),
    )


@pytest.mark.asyncio
async def test_login_creates_new_user(use_case):
    request = LoginRequest(provider="google", code="auth-code", redirect_uri="http://localhost")
    result = await use_case.execute(request)

    assert result.user.email == "new@gmail.com"
    assert result.token.access_token
    assert result.token.refresh_token


@pytest.mark.asyncio
async def test_login_existing_user():
    user_repo = MockUserRepository()
    existing = make_user(
        oauth_provider=OAuthProvider.GOOGLE,
        oauth_subject="existing-sub",
    )
    await user_repo.save(existing)

    uc = LoginWithOAuth(
        auth_providers={
            "google": MockAuthProvider(
                email="existing@gmail.com",
                provider=OAuthProvider.GOOGLE,
                subject="existing-sub",
            ),
        },
        user_repo=user_repo,
        session_repo=MockSessionRepository(),
        token_service=MockTokenService(),
    )

    result = await uc.execute(
        LoginRequest(provider="google", code="code", redirect_uri="http://localhost")
    )
    assert result.user.id == str(existing.id)


@pytest.mark.asyncio
async def test_login_invalid_provider(use_case):
    request = LoginRequest(provider="facebook", code="code", redirect_uri="http://localhost")
    with pytest.raises(ValueError, match="Unsupported provider"):
        await use_case.execute(request)
