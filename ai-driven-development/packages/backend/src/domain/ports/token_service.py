"""Token service port."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.value_objects.token_pair import TokenPair
from src.domain.value_objects.user_id import UserId


class TokenService(ABC):
    @abstractmethod
    def create_token_pair(self, user_id: UserId) -> TokenPair: ...

    @abstractmethod
    def verify_access_token(self, token: str) -> UserId: ...

    @abstractmethod
    def hash_refresh_token(self, token: str) -> str: ...

    @abstractmethod
    def verify_refresh_token_hash(self, token: str, hashed: str) -> bool: ...
