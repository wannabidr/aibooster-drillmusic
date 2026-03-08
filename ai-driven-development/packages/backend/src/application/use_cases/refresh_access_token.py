"""RefreshAccessToken use case."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from src.application.dto.auth_request import RefreshRequest
from src.application.dto.auth_response import TokenResponse
from src.domain.entities.session import Session
from src.domain.ports.session_repository import SessionRepository
from src.domain.ports.token_service import TokenService

REFRESH_TOKEN_DAYS = 30


class RefreshAccessToken:
    def __init__(
        self,
        session_repo: SessionRepository,
        token_service: TokenService,
    ) -> None:
        self._session_repo = session_repo
        self._token_service = token_service

    async def execute(self, request: RefreshRequest) -> TokenResponse:
        token_hash = self._token_service.hash_refresh_token(request.refresh_token)
        session = await self._session_repo.find_by_refresh_hash(token_hash)

        if session is None:
            raise PermissionError("Invalid refresh token")
        if session.revoked:
            raise PermissionError("Session has been revoked")
        if session.is_expired(datetime.now(UTC)):
            raise PermissionError("Session has expired")

        # Rotate: revoke old session, create new token pair + session
        revoked = session.revoke()
        await self._session_repo.save(revoked)

        new_pair = self._token_service.create_token_pair(session.user_id)
        now = datetime.now(UTC)
        new_session = Session(
            id=uuid.uuid4(),
            user_id=session.user_id,
            refresh_token_hash=self._token_service.hash_refresh_token(new_pair.refresh_token),
            expires_at=now + timedelta(days=REFRESH_TOKEN_DAYS),
            created_at=now,
        )
        await self._session_repo.save(new_session)

        return TokenResponse(
            access_token=new_pair.access_token,
            refresh_token=new_pair.refresh_token,
            expires_in=new_pair.expires_in,
        )
