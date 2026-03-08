"""Tests for GoogleOAuthProvider (mocked HTTP)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.domain.value_objects.oauth_provider import OAuthProvider
from src.infrastructure.auth.google_oauth_provider import GoogleOAuthProvider

GOOGLE_JWT_MODULE = "src.infrastructure.auth.google_oauth_provider.jwt"


@pytest.fixture
def google_provider():
    return GoogleOAuthProvider(
        client_id="test-client-id", client_secret="test-secret"
    )


@pytest.mark.asyncio
async def test_exchange_code_success(google_provider):
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"id_token": "fake-jwt-token"}

    mock_jwks_client = MagicMock()
    mock_signing_key = MagicMock()
    mock_signing_key.key = "fake-key"
    mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key

    decoded_claims = {
        "email": "user@gmail.com",
        "name": "Test User",
        "sub": "google-123",
    }

    with (
        patch("httpx.AsyncClient") as mock_client_cls,
        patch(
            f"{GOOGLE_JWT_MODULE}.PyJWKClient",
            return_value=mock_jwks_client,
        ),
        patch(
            f"{GOOGLE_JWT_MODULE}.decode",
            return_value=decoded_claims,
        ),
    ):
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        info = await google_provider.exchange_code(
            "auth-code", "http://localhost/cb"
        )

    assert info.email == "user@gmail.com"
    assert info.provider == OAuthProvider.GOOGLE
    assert info.subject == "google-123"


@pytest.mark.asyncio
async def test_exchange_code_http_error(google_provider):
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("HTTP 400")

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        with pytest.raises(Exception, match="400"):
            await google_provider.exchange_code(
                "bad-code", "http://localhost/cb"
            )
