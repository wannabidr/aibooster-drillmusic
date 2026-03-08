"""PostgreSQL implementation of UserRepository."""

from __future__ import annotations

from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.entities.user import User
from src.domain.ports.user_repository import UserRepository
from src.domain.value_objects.email import Email
from src.domain.value_objects.oauth_provider import OAuthProvider
from src.domain.value_objects.subscription_tier import SubscriptionTier
from src.domain.value_objects.user_id import UserId
from src.infrastructure.persistence.models import UserModel


class PostgresUserRepository(UserRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, user: User) -> None:
        existing = await self._session.get(UserModel, user.id.value)
        if existing:
            existing.email = user.email.value
            existing.display_name = user.display_name
            existing.oauth_provider = user.oauth_provider.value
            existing.oauth_subject = user.oauth_subject
            existing.subscription_tier = user.subscription_tier.value
            existing.has_contributed = user.has_contributed
            existing.stripe_customer_id = user.stripe_customer_id
            existing.deleted_at = user.deleted_at
        else:
            model = UserModel(
                id=user.id.value,
                email=user.email.value,
                display_name=user.display_name,
                oauth_provider=user.oauth_provider.value,
                oauth_subject=user.oauth_subject,
                subscription_tier=user.subscription_tier.value,
                has_contributed=user.has_contributed,
                stripe_customer_id=user.stripe_customer_id,
                created_at=user.created_at,
                deleted_at=user.deleted_at,
            )
            self._session.add(model)
        await self._session.flush()

    async def find_by_id(self, user_id: UserId) -> User | None:
        model = await self._session.get(UserModel, user_id.value)
        return self._to_entity(model) if model else None

    async def find_by_oauth(
        self, provider: OAuthProvider, subject: str
    ) -> User | None:
        stmt = select(UserModel).where(
            UserModel.oauth_provider == provider.value,
            UserModel.oauth_subject == subject,
        )
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def find_by_stripe_customer(self, customer_id: str) -> User | None:
        stmt = select(UserModel).where(
            UserModel.stripe_customer_id == customer_id,
        )
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def delete(self, user_id: UserId) -> None:
        stmt = sa_delete(UserModel).where(UserModel.id == user_id.value)
        await self._session.execute(stmt)
        await self._session.flush()

    @staticmethod
    def _to_entity(model: UserModel) -> User:
        return User(
            id=UserId(model.id),
            email=Email(model.email),
            display_name=model.display_name,
            oauth_provider=OAuthProvider(model.oauth_provider),
            oauth_subject=model.oauth_subject,
            subscription_tier=SubscriptionTier(model.subscription_tier),
            has_contributed=model.has_contributed,
            created_at=model.created_at,
            deleted_at=model.deleted_at,
            stripe_customer_id=model.stripe_customer_id,
        )
