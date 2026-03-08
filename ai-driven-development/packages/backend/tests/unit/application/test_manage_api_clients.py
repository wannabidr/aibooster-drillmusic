"""Tests for ManageApiClients use case."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from src.application.use_cases.manage_api_clients import ManageApiClients, _hash_api_key
from src.domain.entities.api_client import ApiClient
from src.domain.value_objects.api_tier import ApiTier


@pytest.fixture
def mock_repo() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def use_case(mock_repo: AsyncMock) -> ManageApiClients:
    return ManageApiClients(repo=mock_repo)


class TestCreateClient:
    async def test_creates_client_with_correct_tier(
        self, use_case: ManageApiClients, mock_repo: AsyncMock
    ) -> None:
        client, raw_key = await use_case.create_client("Test Label", ApiTier.PRO)

        assert client.organization == "Test Label"
        assert client.tier == ApiTier.PRO
        assert client.is_active is True
        assert raw_key.startswith("aidj_b2b_")
        mock_repo.save.assert_called_once_with(client)

    async def test_key_hash_matches(
        self, use_case: ManageApiClients, mock_repo: AsyncMock
    ) -> None:
        client, raw_key = await use_case.create_client("Org", ApiTier.BASIC)
        assert client.api_key_hash == _hash_api_key(raw_key)

    async def test_unique_keys_per_call(
        self, use_case: ManageApiClients, mock_repo: AsyncMock
    ) -> None:
        _, key1 = await use_case.create_client("A", ApiTier.BASIC)
        _, key2 = await use_case.create_client("B", ApiTier.BASIC)
        assert key1 != key2


class TestRotateKey:
    async def test_rotate_returns_new_key(
        self, use_case: ManageApiClients, mock_repo: AsyncMock
    ) -> None:
        existing = ApiClient(
            client_id=uuid.uuid4(),
            organization="Test",
            api_key_hash="old_hash",
            tier=ApiTier.BASIC,
            is_active=True,
            created_at=datetime.now(UTC),
        )
        mock_repo.find_by_id.return_value = existing

        new_key = await use_case.rotate_key(existing.client_id)

        assert new_key.startswith("aidj_b2b_")
        mock_repo.update.assert_called_once()
        updated_client = mock_repo.update.call_args[0][0]
        assert updated_client.api_key_hash == _hash_api_key(new_key)

    async def test_rotate_not_found_raises(
        self, use_case: ManageApiClients, mock_repo: AsyncMock
    ) -> None:
        mock_repo.find_by_id.return_value = None
        with pytest.raises(ValueError, match="not found"):
            await use_case.rotate_key(uuid.uuid4())

    async def test_rotate_deactivated_raises(
        self, use_case: ManageApiClients, mock_repo: AsyncMock
    ) -> None:
        inactive = ApiClient(
            client_id=uuid.uuid4(),
            organization="Test",
            api_key_hash="hash",
            tier=ApiTier.BASIC,
            is_active=False,
            created_at=datetime.now(UTC),
        )
        mock_repo.find_by_id.return_value = inactive
        with pytest.raises(ValueError, match="deactivated"):
            await use_case.rotate_key(inactive.client_id)


class TestDeactivateClient:
    async def test_deactivate_success(
        self, use_case: ManageApiClients, mock_repo: AsyncMock
    ) -> None:
        existing = ApiClient(
            client_id=uuid.uuid4(),
            organization="Test",
            api_key_hash="hash",
            tier=ApiTier.PRO,
            is_active=True,
            created_at=datetime.now(UTC),
        )
        mock_repo.find_by_id.return_value = existing

        await use_case.deactivate_client(existing.client_id)

        mock_repo.update.assert_called_once()
        updated = mock_repo.update.call_args[0][0]
        assert updated.is_active is False

    async def test_deactivate_not_found_raises(
        self, use_case: ManageApiClients, mock_repo: AsyncMock
    ) -> None:
        mock_repo.find_by_id.return_value = None
        with pytest.raises(ValueError, match="not found"):
            await use_case.deactivate_client(uuid.uuid4())
