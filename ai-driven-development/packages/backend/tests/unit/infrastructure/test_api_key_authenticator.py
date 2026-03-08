"""Tests for ApiKeyAuthenticator."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.domain.entities.api_client import ApiClient
from src.domain.value_objects.api_tier import ApiTier
from src.infrastructure.auth.api_key_authenticator import ApiKeyAuthenticator


@pytest.fixture
def mock_repo() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def authenticator(mock_repo: AsyncMock) -> ApiKeyAuthenticator:
    return ApiKeyAuthenticator(repo=mock_repo)


class TestAuthenticate:
    async def test_valid_key_returns_client(
        self, authenticator: ApiKeyAuthenticator, mock_repo: AsyncMock
    ) -> None:
        raw_key = "aidj_b2b_test_key_123"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        client = ApiClient(
            client_id=uuid.uuid4(),
            organization="Test",
            api_key_hash=key_hash,
            tier=ApiTier.BASIC,
            is_active=True,
            created_at=datetime.now(UTC),
        )
        mock_repo.find_by_api_key_hash.return_value = client

        result = await authenticator.authenticate(raw_key)

        assert result == client
        mock_repo.find_by_api_key_hash.assert_called_once_with(key_hash)

    async def test_invalid_key_returns_none(
        self, authenticator: ApiKeyAuthenticator, mock_repo: AsyncMock
    ) -> None:
        mock_repo.find_by_api_key_hash.return_value = None

        result = await authenticator.authenticate("bad_key")

        assert result is None

    async def test_deactivated_client_returns_none(
        self, authenticator: ApiKeyAuthenticator, mock_repo: AsyncMock
    ) -> None:
        raw_key = "aidj_b2b_inactive"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        client = ApiClient(
            client_id=uuid.uuid4(),
            organization="Inactive Org",
            api_key_hash=key_hash,
            tier=ApiTier.PRO,
            is_active=False,
            created_at=datetime.now(UTC),
        )
        mock_repo.find_by_api_key_hash.return_value = client

        result = await authenticator.authenticate(raw_key)

        assert result is None
