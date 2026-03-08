"""Auth middleware -- FastAPI dependency for JWT authentication."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException

from src.domain.entities.user import User
from src.domain.ports.token_service import TokenService
from src.domain.ports.user_repository import UserRepository


def _get_token_service() -> TokenService:
    """Override this dependency in app factory."""
    raise NotImplementedError("Override in app factory")


def _get_user_repo() -> UserRepository:
    """Override this dependency in app factory."""
    raise NotImplementedError("Override in app factory")


async def get_current_user(
    authorization: str = Header(default=""),
    token_service: TokenService = Depends(_get_token_service),
    user_repo: UserRepository = Depends(_get_user_repo),
) -> User:
    """Extract and validate Bearer token, return the current User."""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token = authorization[7:]
    try:
        user_id = token_service.verify_access_token(token)
    except (PermissionError, ValueError) as e:
        raise HTTPException(status_code=401, detail=str(e)) from e

    user = await user_repo.find_by_id(user_id)
    if user is None or user.is_deleted:
        raise HTTPException(status_code=401, detail="User not found")

    return user
