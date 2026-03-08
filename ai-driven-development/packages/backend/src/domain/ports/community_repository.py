"""Community repository port -- for anonymous transitions and scores."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.domain.entities.anonymous_transition import AnonymousTransition
from src.domain.entities.community_score import CommunityScore


class CommunityRepository(ABC):
    @abstractmethod
    async def save_transitions(self, transitions: list[AnonymousTransition]) -> None: ...

    @abstractmethod
    async def get_scores_for_track(
        self, track_fingerprint: str, limit: int = 20
    ) -> list[CommunityScore]: ...

    @abstractmethod
    async def get_score(
        self, track_a_fingerprint: str, track_b_fingerprint: str
    ) -> CommunityScore | None: ...

    @abstractmethod
    async def increment_scores(self, transitions: list[AnonymousTransition]) -> None: ...
