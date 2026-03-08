"""User repository port."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.user import User
from src.domain.value_objects.oauth_provider import OAuthProvider
from src.domain.value_objects.user_id import UserId


class UserRepository(ABC):
    @abstractmethod
    async def save(self, user: User) -> None: ...

    @abstractmethod
    async def find_by_id(self, user_id: UserId) -> User | None: ...

    @abstractmethod
    async def find_by_oauth(self, provider: OAuthProvider, subject: str) -> User | None: ...

    @abstractmethod
    async def find_by_stripe_customer(self, customer_id: str) -> User | None: ...

    @abstractmethod
    async def delete(self, user_id: UserId) -> None: ...
