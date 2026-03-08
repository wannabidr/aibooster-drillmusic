"""Tests for AppleOAuthProvider (mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from src.domain.value_objects.oauth_provider import OAuthProvider
from src.infrastructure.auth.apple_oauth_provider import AppleOAuthProvider

APPLE_JWT_MODULE = "src.infrastructure.auth.apple_oauth_provider.jwt"


@pytest.fixture
def apple_provider():
    return AppleOAuthProvider(client_id="com.aidj.app")


@pytest.mark.asyncio
async def test_exchange_code_success(apple_provider):
    mock_jwks_client = MagicMock()
    mock_signing_key = MagicMock()
    mock_signing_key.key = "fake-key"
    mock_jwks_client.get_signing_key_from_jwt.return_value = mock_signing_key

    decoded_claims = {"email": "user@icloud.com", "sub": "apple-001"}

    with (
        patch(f"{APPLE_JWT_MODULE}.PyJWKClient", return_value=mock_jwks_client),
        patch(f"{APPLE_JWT_MODULE}.decode", return_value=decoded_claims),
    ):
        info = await apple_provider.exchange_code("identity-token-jwt", "")

    assert info.email == "user@icloud.com"
    assert info.provider == OAuthProvider.APPLE
    assert info.subject == "apple-001"


@pytest.mark.asyncio
async def test_exchange_code_invalid_token(apple_provider):
    mock_jwks_client = MagicMock()
    mock_jwks_client.get_signing_key_from_jwt.side_effect = Exception("Invalid JWT")

    with patch(
        f"{APPLE_JWT_MODULE}.PyJWKClient",
        return_value=mock_jwks_client,
    ), pytest.raises(Exception, match="Invalid JWT"):
        await apple_provider.exchange_code("bad-token", "")
