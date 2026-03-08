"""Session repository port."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.session import Session
from src.domain.value_objects.user_id import UserId


class SessionRepository(ABC):
    @abstractmethod
    async def save(self, session: Session) -> None: ...

    @abstractmethod
    async def find_by_refresh_hash(self, token_hash: str) -> Session | None: ...

    @abstractmethod
    async def revoke_all_for_user(self, user_id: UserId) -> None: ...
