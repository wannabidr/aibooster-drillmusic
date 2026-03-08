"""PostgreSQL community repository implementation."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.entities.anonymous_transition import AnonymousTransition
from src.domain.entities.community_score import CommunityScore
from src.domain.ports.community_repository import CommunityRepository
from src.infrastructure.persistence.models import (
    AnonymousTransitionModel,
    CommunityScoreModel,
)


class PostgresCommunityRepository(CommunityRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save_transitions(self, transitions: list[AnonymousTransition]) -> None:
        models = [
            AnonymousTransitionModel(
                id=t.id,
                track_a_fingerprint=t.track_a_fingerprint,
                track_b_fingerprint=t.track_b_fingerprint,
                contributed_at=t.contributed_at,
            )
            for t in transitions
        ]
        self._session.add_all(models)
        await self._session.flush()

    async def get_scores_for_track(
        self, track_fingerprint: str, limit: int = 20
    ) -> list[CommunityScore]:
        stmt = (
            select(CommunityScoreModel)
            .where(CommunityScoreModel.track_a_fingerprint == track_fingerprint)
            .order_by(CommunityScoreModel.frequency.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return [self._to_entity(m) for m in result.scalars().all()]

    async def get_score(
        self, track_a_fingerprint: str, track_b_fingerprint: str
    ) -> CommunityScore | None:
        model = await self._session.get(
            CommunityScoreModel,
            (track_a_fingerprint, track_b_fingerprint),
        )
        if model is None:
            return None
        return self._to_entity(model)

    async def increment_scores(self, transitions: list[AnonymousTransition]) -> None:
        for t in transitions:
            model = await self._session.get(
                CommunityScoreModel,
                (t.track_a_fingerprint, t.track_b_fingerprint),
            )
            if model:
                model.frequency += 1
            else:
                self._session.add(
                    CommunityScoreModel(
                        track_a_fingerprint=t.track_a_fingerprint,
                        track_b_fingerprint=t.track_b_fingerprint,
                        frequency=1,
                    )
                )
        await self._session.flush()

    @staticmethod
    def _to_entity(model: CommunityScoreModel) -> CommunityScore:
        return CommunityScore(
            track_a_fingerprint=model.track_a_fingerprint,
            track_b_fingerprint=model.track_b_fingerprint,
            frequency=model.frequency,
        )
