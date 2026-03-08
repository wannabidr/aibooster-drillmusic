"""ApiClient entity -- B2B API consumer."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from datetime import datetime

from src.domain.value_objects.api_tier import ApiTier


@dataclass(frozen=True)
class ApiClient:
    client_id: uuid.UUID
    organization: str
    api_key_hash: str
    tier: ApiTier
    is_active: bool
    created_at: datetime

    def deactivate(self) -> ApiClient:
        return replace(self, is_active=False)

    def upgrade_tier(self, tier: ApiTier) -> ApiClient:
        return replace(self, tier=tier)

    def rotate_key(self, new_key_hash: str) -> ApiClient:
        return replace(self, api_key_hash=new_key_hash)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ApiClient):
            return NotImplemented
        return self.client_id == other.client_id

    def __hash__(self) -> int:
        return hash(self.client_id)
