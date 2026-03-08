"""Manage B2B API clients use case."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from datetime import UTC, datetime

from src.domain.entities.api_client import ApiClient
from src.domain.ports.api_client_repository import ApiClientRepository
from src.domain.value_objects.api_tier import ApiTier


def _hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()


class ManageApiClients:
    def __init__(self, repo: ApiClientRepository) -> None:
        self._repo = repo

    async def create_client(self, organization: str, tier: ApiTier) -> tuple[ApiClient, str]:
        """Create a new API client. Returns (client, raw_api_key)."""
        raw_key = f"aidj_b2b_{secrets.token_urlsafe(32)}"
        client = ApiClient(
            client_id=uuid.uuid4(),
            organization=organization,
            api_key_hash=_hash_api_key(raw_key),
            tier=tier,
            is_active=True,
            created_at=datetime.now(UTC),
        )
        await self._repo.save(client)
        return client, raw_key

    async def rotate_key(self, client_id: uuid.UUID) -> str:
        """Rotate the API key for a client. Returns the new raw key."""
        client = await self._repo.find_by_id(client_id)
        if client is None:
            raise ValueError(f"API client {client_id} not found")
        if not client.is_active:
            raise ValueError("Cannot rotate key for deactivated client")

        raw_key = f"aidj_b2b_{secrets.token_urlsafe(32)}"
        updated = client.rotate_key(_hash_api_key(raw_key))
        await self._repo.update(updated)
        return raw_key

    async def deactivate_client(self, client_id: uuid.UUID) -> None:
        """Deactivate a client, revoking API access."""
        client = await self._repo.find_by_id(client_id)
        if client is None:
            raise ValueError(f"API client {client_id} not found")
        await self._repo.update(client.deactivate())
