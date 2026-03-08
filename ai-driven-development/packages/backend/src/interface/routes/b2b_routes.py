"""B2B API routes -- anonymized trend data for labels and promoters.

All endpoints require API key authentication via X-Api-Key header.
Rate limits are enforced per API tier (basic/pro/enterprise).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from src.application.use_cases.query_trends import QueryTrends
from src.domain.entities.api_client import ApiClient
from src.domain.value_objects.api_tier import ApiTier
from src.interface.middleware.api_key_middleware import get_api_client

router = APIRouter(prefix="/api/v1/b2b", tags=["B2B"])


# --- Response Models ---


class GenreTrendItemModel(BaseModel):
    genre: str
    play_count: int
    change_pct: float


class GenreTrendResponse(BaseModel):
    region: str | None
    days: int
    trends: list[GenreTrendItemModel]


class BpmBucketModel(BaseModel):
    bpm_min: float
    bpm_max: float
    count: int


class BpmDistributionResponse(BaseModel):
    genre: str | None
    days: int
    buckets: list[BpmBucketModel]
    mean_bpm: float
    median_bpm: float


class AnonymizedTrackModel(BaseModel):
    fingerprint: str
    genre: str
    bpm: float
    key: str
    play_count: int


class TopTracksResponse(BaseModel):
    tracks: list[AnonymizedTrackModel]


class TrendingTransitionModel(BaseModel):
    track_a_fingerprint: str
    track_b_fingerprint: str
    frequency: int
    genre: str


class TrendingTransitionsResponse(BaseModel):
    transitions: list[TrendingTransitionModel]


# --- Dependency stubs ---


def _get_query_trends() -> QueryTrends:
    """Override this dependency in app factory."""
    raise HTTPException(status_code=501, detail="B2B API not configured")


# --- Tier access helpers ---


def _require_tier(client: ApiClient, minimum: ApiTier) -> None:
    """Raise 403 if client tier is below the required minimum."""
    tier_order = [ApiTier.BASIC, ApiTier.PRO, ApiTier.ENTERPRISE]
    if tier_order.index(client.tier) < tier_order.index(minimum):
        raise HTTPException(
            status_code=403,
            detail=f"This endpoint requires {minimum.value} tier or above. "
            f"Your tier: {client.tier.value}",
        )


# --- Endpoints ---


@router.get("/trends/genre", response_model=GenreTrendResponse)
async def genre_trends(
    region: str | None = Query(None, description="Region filter (e.g. 'us', 'eu')"),
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    client: ApiClient = Depends(get_api_client),
    use_case: QueryTrends = Depends(_get_query_trends),
) -> GenreTrendResponse:
    """Get genre popularity trends. Available to all B2B tiers."""
    result = await use_case.genre_trends(region, days)
    return GenreTrendResponse(
        region=result.region,
        days=result.days,
        trends=[
            GenreTrendItemModel(
                genre=t.genre,
                play_count=t.play_count,
                change_pct=t.change_pct,
            )
            for t in result.trends
        ],
    )


@router.get("/trends/bpm", response_model=BpmDistributionResponse)
async def bpm_distribution(
    genre: str | None = Query(None, description="Filter by genre"),
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    client: ApiClient = Depends(get_api_client),
    use_case: QueryTrends = Depends(_get_query_trends),
) -> BpmDistributionResponse:
    """Get BPM distribution data. Available to all B2B tiers."""
    result = await use_case.bpm_distribution(genre, days)
    return BpmDistributionResponse(
        genre=result.genre,
        days=result.days,
        buckets=[
            BpmBucketModel(bpm_min=b.bpm_min, bpm_max=b.bpm_max, count=b.count)
            for b in result.buckets
        ],
        mean_bpm=result.mean_bpm,
        median_bpm=result.median_bpm,
    )


@router.get("/trends/tracks", response_model=TopTracksResponse)
async def top_tracks(
    genre: str | None = Query(None, description="Filter by genre"),
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    client: ApiClient = Depends(get_api_client),
    use_case: QueryTrends = Depends(_get_query_trends),
) -> TopTracksResponse:
    """Get top played tracks (anonymized). Requires PRO tier or above."""
    _require_tier(client, ApiTier.PRO)
    result = await use_case.top_tracks(genre, days, limit)
    return TopTracksResponse(
        tracks=[
            AnonymizedTrackModel(
                fingerprint=t.fingerprint,
                genre=t.genre,
                bpm=t.bpm,
                key=t.key,
                play_count=t.play_count,
            )
            for t in result
        ],
    )


@router.get("/trends/transitions", response_model=TrendingTransitionsResponse)
async def trending_transitions(
    genre: str | None = Query(None, description="Filter by genre"),
    days: int = Query(30, ge=1, le=365, description="Number of days to look back"),
    limit: int = Query(20, ge=1, le=100, description="Max results"),
    client: ApiClient = Depends(get_api_client),
    use_case: QueryTrends = Depends(_get_query_trends),
) -> TrendingTransitionsResponse:
    """Get trending track transitions. Requires ENTERPRISE tier."""
    _require_tier(client, ApiTier.ENTERPRISE)
    result = await use_case.trending_transitions(genre, days, limit)
    return TrendingTransitionsResponse(
        transitions=[
            TrendingTransitionModel(
                track_a_fingerprint=t.track_a_fingerprint,
                track_b_fingerprint=t.track_b_fingerprint,
                frequency=t.frequency,
                genre=t.genre,
            )
            for t in result
        ],
    )
