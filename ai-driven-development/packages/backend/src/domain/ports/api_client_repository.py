"""ApiClient repository port."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

from src.domain.entities.api_client import ApiClient


class ApiClientRepository(ABC):
    @abstractmethod
    async def find_by_api_key_hash(self, api_key_hash: str) -> ApiClient | None: ...

    @abstractmethod
    async def find_by_id(self, client_id: uuid.UUID) -> ApiClient | None: ...

    @abstractmethod
    async def save(self, client: ApiClient) -> None: ...

    @abstractmethod
    async def update(self, client: ApiClient) -> None: ...
