"""DeleteAccount use case -- right-to-deletion (GDPR)."""

from __future__ import annotations

from src.domain.ports.session_repository import SessionRepository
from src.domain.ports.user_repository import UserRepository
from src.domain.value_objects.user_id import UserId


class DeleteAccount:
    def __init__(
        self,
        user_repo: UserRepository,
        session_repo: SessionRepository,
    ) -> None:
        self._user_repo = user_repo
        self._session_repo = session_repo

    async def execute(self, user_id: UserId) -> None:
        await self._session_repo.revoke_all_for_user(user_id)
        await self._user_repo.delete(user_id)
