"""Tests for ApiClient entity and ApiTier value object."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from src.domain.entities.api_client import ApiClient
from src.domain.value_objects.api_tier import ApiTier


@pytest.fixture
def client() -> ApiClient:
    return ApiClient(
        client_id=uuid.uuid4(),
        organization="Test Label",
        api_key_hash="abc123hash",
        tier=ApiTier.BASIC,
        is_active=True,
        created_at=datetime.now(UTC),
    )


class TestApiTier:
    def test_tier_values(self) -> None:
        assert ApiTier.BASIC == "basic"
        assert ApiTier.PRO == "pro"
        assert ApiTier.ENTERPRISE == "enterprise"

    def test_rate_limits(self) -> None:
        assert ApiTier.BASIC.rate_limit == 100
        assert ApiTier.PRO.rate_limit == 1000
        assert ApiTier.ENTERPRISE.rate_limit == 5000

    def test_tier_from_string(self) -> None:
        assert ApiTier("basic") == ApiTier.BASIC
        assert ApiTier("pro") == ApiTier.PRO
        assert ApiTier("enterprise") == ApiTier.ENTERPRISE

    def test_invalid_tier_raises(self) -> None:
        with pytest.raises(ValueError):
            ApiTier("invalid")


class TestApiClient:
    def test_create_client(self, client: ApiClient) -> None:
        assert client.organization == "Test Label"
        assert client.tier == ApiTier.BASIC
        assert client.is_active is True

    def test_deactivate(self, client: ApiClient) -> None:
        deactivated = client.deactivate()
        assert deactivated.is_active is False
        assert deactivated.client_id == client.client_id

    def test_upgrade_tier(self, client: ApiClient) -> None:
        upgraded = client.upgrade_tier(ApiTier.PRO)
        assert upgraded.tier == ApiTier.PRO
        assert upgraded.client_id == client.client_id

    def test_rotate_key(self, client: ApiClient) -> None:
        rotated = client.rotate_key("new_hash")
        assert rotated.api_key_hash == "new_hash"
        assert rotated.client_id == client.client_id

    def test_frozen_immutability(self, client: ApiClient) -> None:
        with pytest.raises(AttributeError):
            client.organization = "Other"  # type: ignore[misc]

    def test_equality_by_id(self) -> None:
        cid = uuid.uuid4()
        now = datetime.now(UTC)
        c1 = ApiClient(cid, "A", "hash1", ApiTier.BASIC, True, now)
        c2 = ApiClient(cid, "B", "hash2", ApiTier.PRO, False, now)
        assert c1 == c2

    def test_inequality(self) -> None:
        now = datetime.now(UTC)
        c1 = ApiClient(uuid.uuid4(), "A", "h1", ApiTier.BASIC, True, now)
        c2 = ApiClient(uuid.uuid4(), "A", "h1", ApiTier.BASIC, True, now)
        assert c1 != c2

    def test_hash_by_id(self) -> None:
        cid = uuid.uuid4()
        now = datetime.now(UTC)
        c1 = ApiClient(cid, "A", "h1", ApiTier.BASIC, True, now)
        c2 = ApiClient(cid, "B", "h2", ApiTier.PRO, False, now)
        assert hash(c1) == hash(c2)
        assert len({c1, c2}) == 1
