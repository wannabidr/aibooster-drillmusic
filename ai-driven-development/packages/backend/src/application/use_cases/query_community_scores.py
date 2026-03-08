"""QueryCommunityScores use case."""

from __future__ import annotations

from src.application.dto.auth_response import ScoreResponse
from src.domain.ports.community_repository import CommunityRepository


class QueryCommunityScores:
    def __init__(self, community_repo: CommunityRepository) -> None:
        self._community_repo = community_repo

    async def execute(self, track_fingerprint: str, limit: int = 20) -> list[ScoreResponse]:
        scores = await self._community_repo.get_scores_for_track(track_fingerprint, limit)
        return [
            ScoreResponse(
                track_a_fingerprint=s.track_a_fingerprint,
                track_b_fingerprint=s.track_b_fingerprint,
                frequency=s.frequency,
            )
            for s in scores
        ]
