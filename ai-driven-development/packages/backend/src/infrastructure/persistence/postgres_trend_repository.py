"""PostgreSQL trend aggregator implementation.

Aggregates anonymized community data for B2B API consumption.
All queries operate on the existing anonymous_transitions and community_scores tables.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import Float, Integer, String, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.domain.ports.trend_aggregator import (
    AnonymizedTrack,
    BpmBucket,
    BpmDistribution,
    GenreTrendItem,
    GenreTrendReport,
    TrendAggregator,
    TrendingTransition,
)
from src.infrastructure.persistence.models import (
    AnonymousTransitionModel,
    CommunityScoreModel,
    TrendCacheModel,
)


class PostgresTrendRepository(TrendAggregator):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def genre_trends(self, region: str | None, days: int) -> GenreTrendReport:
        cutoff = datetime.now(UTC) - timedelta(days=days)
        prev_cutoff = cutoff - timedelta(days=days)

        # Current period counts by genre
        current_stmt = (
            select(
                TrendCacheModel.genre,
                func.sum(TrendCacheModel.play_count).label("total"),
            )
            .where(TrendCacheModel.bucket_time >= cutoff)
            .group_by(TrendCacheModel.genre)
            .order_by(func.sum(TrendCacheModel.play_count).desc())
        )
        if region:
            current_stmt = current_stmt.where(TrendCacheModel.region == region)

        # Previous period counts for change calculation
        prev_stmt = (
            select(
                TrendCacheModel.genre,
                func.sum(TrendCacheModel.play_count).label("total"),
            )
            .where(
                TrendCacheModel.bucket_time >= prev_cutoff,
                TrendCacheModel.bucket_time < cutoff,
            )
            .group_by(TrendCacheModel.genre)
        )
        if region:
            prev_stmt = prev_stmt.where(TrendCacheModel.region == region)

        current_result = await self._session.execute(current_stmt)
        current_rows = {r.genre: r.total for r in current_result}

        prev_result = await self._session.execute(prev_stmt)
        prev_rows = {r.genre: r.total for r in prev_result}

        trends = []
        for genre, count in current_rows.items():
            prev_count = prev_rows.get(genre, 0)
            change = ((count - prev_count) / max(prev_count, 1)) * 100.0
            trends.append(
                GenreTrendItem(genre=genre, play_count=count, change_pct=round(change, 1))
            )

        trends.sort(key=lambda t: t.play_count, reverse=True)

        return GenreTrendReport(
            region=region or "global",
            days=days,
            trends=trends,
        )

    async def bpm_distribution(self, genre: str | None, days: int) -> BpmDistribution:
        cutoff = datetime.now(UTC) - timedelta(days=days)

        stmt = select(
            TrendCacheModel.bpm,
            TrendCacheModel.play_count,
        ).where(
            TrendCacheModel.bucket_time >= cutoff,
            TrendCacheModel.bpm.is_not(None),
        )
        if genre:
            stmt = stmt.where(TrendCacheModel.genre == genre)

        result = await self._session.execute(stmt)
        rows = list(result)

        if not rows:
            return BpmDistribution(
                genre=genre or "all",
                days=days,
                buckets=[],
                mean_bpm=0.0,
                median_bpm=0.0,
            )

        # Build BPM buckets (5 BPM width)
        bpm_counts: dict[int, int] = {}
        all_bpms: list[float] = []
        total_weight = 0

        for row in rows:
            bpm = row.bpm
            count = row.play_count
            bucket_key = int(bpm // 5) * 5
            bpm_counts[bucket_key] = bpm_counts.get(bucket_key, 0) + count
            all_bpms.extend([bpm] * count)
            total_weight += count

        buckets = sorted(
            [
                BpmBucket(bpm_min=float(k), bpm_max=float(k + 5), count=v)
                for k, v in bpm_counts.items()
            ],
            key=lambda b: b.bpm_min,
        )

        all_bpms.sort()
        mean_bpm = sum(all_bpms) / len(all_bpms) if all_bpms else 0.0
        mid = len(all_bpms) // 2
        median_bpm = (
            all_bpms[mid]
            if len(all_bpms) % 2
            else (all_bpms[mid - 1] + all_bpms[mid]) / 2.0
        ) if all_bpms else 0.0

        return BpmDistribution(
            genre=genre or "all",
            days=days,
            buckets=buckets,
            mean_bpm=round(mean_bpm, 1),
            median_bpm=round(median_bpm, 1),
        )

    async def top_tracks(
        self, genre: str | None, days: int, limit: int
    ) -> list[AnonymizedTrack]:
        cutoff = datetime.now(UTC) - timedelta(days=days)

        stmt = (
            select(
                TrendCacheModel.fingerprint,
                TrendCacheModel.genre,
                TrendCacheModel.bpm,
                TrendCacheModel.key,
                func.sum(TrendCacheModel.play_count).label("total"),
            )
            .where(TrendCacheModel.bucket_time >= cutoff)
            .group_by(
                TrendCacheModel.fingerprint,
                TrendCacheModel.genre,
                TrendCacheModel.bpm,
                TrendCacheModel.key,
            )
            .order_by(func.sum(TrendCacheModel.play_count).desc())
            .limit(limit)
        )
        if genre:
            stmt = stmt.where(TrendCacheModel.genre == genre)

        result = await self._session.execute(stmt)
        return [
            AnonymizedTrack(
                fingerprint=r.fingerprint,
                genre=r.genre or "unknown",
                bpm=r.bpm or 0.0,
                key=r.key or "unknown",
                play_count=r.total,
            )
            for r in result
        ]

    async def trending_transitions(
        self, genre: str | None, days: int, limit: int
    ) -> list[TrendingTransition]:
        cutoff = datetime.now(UTC) - timedelta(days=days)

        stmt = (
            select(CommunityScoreModel)
            .where(CommunityScoreModel.updated_at >= cutoff)
            .order_by(CommunityScoreModel.frequency.desc())
            .limit(limit)
        )

        result = await self._session.execute(stmt)
        return [
            TrendingTransition(
                track_a_fingerprint=r.track_a_fingerprint,
                track_b_fingerprint=r.track_b_fingerprint,
                frequency=r.frequency,
                genre="mixed",
            )
            for r in result.scalars().all()
        ]
