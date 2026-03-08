"""Community routes -- sync anonymized history and query scores."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.application.dto.auth_request import TransitionData
from src.application.use_cases.check_feature_access import CheckFeatureAccess
from src.application.use_cases.query_community_scores import QueryCommunityScores
from src.application.use_cases.sync_mix_history import SyncMixHistory
from src.domain.entities.user import User
from src.interface.middleware.auth_middleware import get_current_user

router = APIRouter()


class TransitionItem(BaseModel):
    track_a_fingerprint: str
    track_b_fingerprint: str


class SyncBody(BaseModel):
    transitions: list[TransitionItem]


class SyncResponseModel(BaseModel):
    synced_count: int
    has_contributed: bool


class ScoreItem(BaseModel):
    track_a_fingerprint: str
    track_b_fingerprint: str
    frequency: int


class ScoresResponseModel(BaseModel):
    scores: list[ScoreItem]


def _get_sync_use_case() -> SyncMixHistory:
    """Override this dependency in app factory."""
    raise HTTPException(status_code=501, detail="Sync not configured")


def _get_scores_use_case() -> QueryCommunityScores:
    """Override this dependency in app factory."""
    raise HTTPException(status_code=501, detail="Scores not configured")


@router.post("/sync", response_model=SyncResponseModel)
async def sync_history(
    body: SyncBody,
    user: User = Depends(get_current_user),
    use_case: SyncMixHistory = Depends(_get_sync_use_case),
) -> SyncResponseModel:
    """Upload anonymized transition data and mark user as contributor."""
    transitions = [
        TransitionData(
            track_a_fingerprint=t.track_a_fingerprint,
            track_b_fingerprint=t.track_b_fingerprint,
        )
        for t in body.transitions
    ]

    count = await use_case.execute(user, transitions)
    return SyncResponseModel(synced_count=count, has_contributed=True)


@router.get("/scores", response_model=ScoresResponseModel)
async def get_scores(
    track_fingerprint: str = Query(..., description="Track fingerprint to query"),
    limit: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    use_case: QueryCommunityScores = Depends(_get_scores_use_case),
) -> ScoresResponseModel:
    """Query community transition scores. Requires contribution (give-to-get)."""
    checker = CheckFeatureAccess()
    if not checker.execute(user, "community_scores"):
        raise HTTPException(
            status_code=403,
            detail="Share your mix history first to access community scores (give-to-get model).",
        )

    results = await use_case.execute(track_fingerprint, limit)
    return ScoresResponseModel(
        scores=[
            ScoreItem(
                track_a_fingerprint=r.track_a_fingerprint,
                track_b_fingerprint=r.track_b_fingerprint,
                frequency=r.frequency,
            )
            for r in results
        ]
    )
