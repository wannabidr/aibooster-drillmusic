"""SyncMixHistory use case -- upload anonymized transition data."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from src.application.dto.auth_request import TransitionData
from src.domain.entities.anonymous_transition import AnonymousTransition
from src.domain.entities.user import User
from src.domain.ports.community_repository import CommunityRepository
from src.domain.ports.user_repository import UserRepository


class SyncMixHistory:
    def __init__(
        self,
        community_repo: CommunityRepository,
        user_repo: UserRepository,
    ) -> None:
        self._community_repo = community_repo
        self._user_repo = user_repo

    async def execute(self, user: User, transitions: list[TransitionData]) -> int:
        if not transitions:
            return 0

        now = datetime.now(UTC)
        anon_transitions = [
            AnonymousTransition(
                id=uuid.uuid4(),
                track_a_fingerprint=t.track_a_fingerprint,
                track_b_fingerprint=t.track_b_fingerprint,
                contributed_at=now,
            )
            for t in transitions
        ]

        await self._community_repo.save_transitions(anon_transitions)
        await self._community_repo.increment_scores(anon_transitions)

        if not user.has_contributed:
            updated = user.mark_contributed()
            await self._user_repo.save(updated)

        return len(anon_transitions)
