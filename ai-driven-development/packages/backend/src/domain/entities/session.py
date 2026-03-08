"""Session entity."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, replace
from datetime import datetime

from src.domain.value_objects.user_id import UserId


@dataclass(frozen=True)
class Session:
    id: uuid.UUID
    user_id: UserId
    refresh_token_hash: str
    expires_at: datetime
    revoked: bool = False
    created_at: datetime | None = None

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at

    def revoke(self) -> Session:
        return replace(self, revoked=True)
