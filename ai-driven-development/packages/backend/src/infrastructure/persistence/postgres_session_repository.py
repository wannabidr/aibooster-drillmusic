"""PostgreSQL implementation of SessionRepository."""

from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.entities.session import Session
from src.domain.ports.session_repository import SessionRepository
from src.domain.value_objects.user_id import UserId
from src.infrastructure.persistence.models import SessionModel


class PostgresSessionRepository(SessionRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(self, session: Session) -> None:
        existing = await self._session.get(SessionModel, session.id)
        if existing:
            existing.revoked = session.revoked
        else:
            model = SessionModel(
                id=session.id,
                user_id=session.user_id.value,
                refresh_token_hash=session.refresh_token_hash,
                expires_at=session.expires_at,
                revoked=session.revoked,
                created_at=session.created_at,
            )
            self._session.add(model)
        await self._session.flush()

    async def find_by_refresh_hash(self, token_hash: str) -> Session | None:
        stmt = select(SessionModel).where(
            SessionModel.refresh_token_hash == token_hash,
        )
        result = await self._session.execute(stmt)
        model = result.scalar_one_or_none()
        return self._to_entity(model) if model else None

    async def revoke_all_for_user(self, user_id: UserId) -> None:
        stmt = (
            update(SessionModel)
            .where(
                SessionModel.user_id == user_id.value,
                SessionModel.revoked.is_(False),
            )
            .values(revoked=True)
        )
        await self._session.execute(stmt)
        await self._session.flush()

    @staticmethod
    def _to_entity(model: SessionModel) -> Session:
        return Session(
            id=model.id,
            user_id=UserId(model.user_id),
            refresh_token_hash=model.refresh_token_hash,
            expires_at=model.expires_at,
            revoked=model.revoked,
            created_at=model.created_at,
        )
