"""Logout use case -- revoke a session."""

from __future__ import annotations

from src.domain.ports.session_repository import SessionRepository
from src.domain.ports.token_service import TokenService


class Logout:
    def __init__(
        self,
        session_repo: SessionRepository,
        token_service: TokenService,
    ) -> None:
        self._session_repo = session_repo
        self._token_service = token_service

    async def execute(self, refresh_token: str) -> None:
        token_hash = self._token_service.hash_refresh_token(refresh_token)
        session = await self._session_repo.find_by_refresh_hash(token_hash)

        if session is None or session.revoked:
            return

        revoked = session.revoke()
        await self._session_repo.save(revoked)
