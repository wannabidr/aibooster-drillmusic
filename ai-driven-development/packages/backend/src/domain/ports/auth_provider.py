"""OAuth provider port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.domain.value_objects.oauth_provider import OAuthProvider


@dataclass(frozen=True)
class OAuthUserInfo:
    email: str
    name: str | None
    provider: OAuthProvider
    subject: str


class AuthProvider(ABC):
    @abstractmethod
    async def exchange_code(self, code: str, redirect_uri: str) -> OAuthUserInfo: ...
