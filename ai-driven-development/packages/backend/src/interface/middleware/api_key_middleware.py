"""API key authentication middleware for B2B routes."""

from __future__ import annotations

from fastapi import Depends, Header, HTTPException

from src.domain.entities.api_client import ApiClient
from src.infrastructure.auth.api_key_authenticator import ApiKeyAuthenticator


def _get_authenticator() -> ApiKeyAuthenticator:
    """Override this dependency in app factory."""
    raise HTTPException(status_code=501, detail="B2B API not configured")


async def get_api_client(
    x_api_key: str = Header(default=""),
    authenticator: ApiKeyAuthenticator = Depends(_get_authenticator),
) -> ApiClient:
    """Extract and validate X-Api-Key header, return the API client."""
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing X-Api-Key header",
        )

    client = await authenticator.authenticate(x_api_key)
    if client is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or deactivated API key",
        )

    return client
