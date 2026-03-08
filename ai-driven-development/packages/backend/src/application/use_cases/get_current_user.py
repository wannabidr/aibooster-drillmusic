"""GetCurrentUser use case -- resolve token to User."""

from __future__ import annotations

from src.domain.entities.user import User
from src.domain.ports.token_service import TokenService
from src.domain.ports.user_repository import UserRepository


class GetCurrentUser:
    def __init__(
        self,
        token_service: TokenService,
        user_repo: UserRepository,
    ) -> None:
        self._token_service = token_service
        self._user_repo = user_repo

    async def execute(self, access_token: str) -> User:
        user_id = self._token_service.verify_access_token(access_token)
        user = await self._user_repo.find_by_id(user_id)
        if user is None or user.is_deleted:
            raise PermissionError("User not found")
        return user
