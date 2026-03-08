"""PostgreSQL API client repository implementation."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.entities.api_client import ApiClient
from src.domain.ports.api_client_repository import ApiClientRepository
from src.domain.value_objects.api_tier import ApiTier
from src.infrastructure.persistence.models import ApiClientModel


class PostgresApiClientRepository(ApiClientRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def find_by_api_key_hash(self, api_key_hash: str) -> ApiClient | None:
        stmt = select(ApiClientModel).where(
            ApiClientModel.api_key_hash == api_key_hash
        )
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)

    async def find_by_id(self, client_id: uuid.UUID) -> ApiClient | None:
        model = await self._session.get(ApiClientModel, client_id)
        if model is None:
            return None
        return self._to_entity(model)

    async def save(self, client: ApiClient) -> None:
        model = ApiClientModel(
            client_id=client.client_id,
            organization=client.organization,
            api_key_hash=client.api_key_hash,
            tier=client.tier.value,
            is_active=client.is_active,
            created_at=client.created_at,
        )
        self._session.add(model)
        await self._session.flush()

    async def update(self, client: ApiClient) -> None:
        model = await self._session.get(ApiClientModel, client.client_id)
        if model is None:
            return
        model.organization = client.organization
        model.api_key_hash = client.api_key_hash
        model.tier = client.tier.value
        model.is_active = client.is_active
        await self._session.flush()

    @staticmethod
    def _to_entity(model: ApiClientModel) -> ApiClient:
        return ApiClient(
            client_id=model.client_id,
            organization=model.organization,
            api_key_hash=model.api_key_hash,
            tier=ApiTier(model.tier),
            is_active=model.is_active,
            created_at=model.created_at,
        )
