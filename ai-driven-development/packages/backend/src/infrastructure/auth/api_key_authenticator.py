"""API key authentication for B2B clients."""

from __future__ import annotations

import hashlib

from src.domain.entities.api_client import ApiClient
from src.domain.ports.api_client_repository import ApiClientRepository


class ApiKeyAuthenticator:
    def __init__(self, repo: ApiClientRepository) -> None:
        self._repo = repo

    async def authenticate(self, api_key: str) -> ApiClient | None:
        """Validate an API key and return the associated client, or None."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        client = await self._repo.find_by_api_key_hash(key_hash)
        if client is None or not client.is_active:
            return None
        return client
