"""Integration tests for SQLiteAnalyticsRepository."""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from src.domain.entities.session_analytics import SessionAnalytics, TrackPlayEvent
from src.infrastructure.persistence.sqlite_analytics_repository import (
    SQLiteAnalyticsRepository,
)


def _make_event(
    minutes_offset: int = 0,
    energy: float = 0.7,
    genre: str = "techno",
    bpm: float = 130.0,
    key: str = "Am",
) -> TrackPlayEvent:
    return TrackPlayEvent(
        track_id=uuid.uuid4(),
        played_at=datetime(2026, 1, 1, 20, 0, tzinfo=UTC) + timedelta(minutes=minutes_offset),
        energy=energy,
        genre=genre,
        bpm=bpm,
        key=key,
    )


def _make_session(
    timestamp: datetime | None = None,
    track_count: int = 3,
) -> SessionAnalytics:
    if timestamp is None:
        timestamp = datetime.now(UTC)
    tracks = [_make_event(minutes_offset=i * 5) for i in range(track_count)]
    return SessionAnalytics(
        session_id=uuid.uuid4(),
        timestamp=timestamp,
        tracks_played=tracks,
        energy_curve=[0.5, 0.7, 0.9][:track_count],
        genre_distribution={"techno": track_count},
        bpm_range=(128.0, 135.0),
        key_transitions=[("Am", "Cm")],
        avg_transition_quality=0.85,
    )


@pytest.fixture
def repo() -> SQLiteAnalyticsRepository:
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    r = SQLiteAnalyticsRepository(path)
    yield r
    os.unlink(path)


class TestSaveAndRetrieve:
    def test_save_and_get_session(self, repo: SQLiteAnalyticsRepository) -> None:
        session = _make_session()
        repo.save_session(session)

        now = datetime.now(UTC)
        past = now - timedelta(hours=1)
        sessions = repo.get_sessions(past, now + timedelta(hours=1))

        assert len(sessions) == 1
        assert sessions[0].session_id == session.session_id
        assert sessions[0].track_count == 3
        assert sessions[0].avg_transition_quality == 0.85

    def test_energy_curve_roundtrip(self, repo: SQLiteAnalyticsRepository) -> None:
        session = _make_session()
        repo.save_session(session)

        now = datetime.now(UTC)
        sessions = repo.get_sessions(now - timedelta(hours=1), now + timedelta(hours=1))

        assert sessions[0].energy_curve == session.energy_curve

    def test_genre_distribution_roundtrip(self, repo: SQLiteAnalyticsRepository) -> None:
        session = _make_session()
        repo.save_session(session)

        now = datetime.now(UTC)
        sessions = repo.get_sessions(now - timedelta(hours=1), now + timedelta(hours=1))

        assert sessions[0].genre_distribution == session.genre_distribution

    def test_key_transitions_roundtrip(self, repo: SQLiteAnalyticsRepository) -> None:
        session = _make_session()
        repo.save_session(session)

        now = datetime.now(UTC)
        sessions = repo.get_sessions(now - timedelta(hours=1), now + timedelta(hours=1))

        assert sessions[0].key_transitions == session.key_transitions

    def test_bpm_range_roundtrip(self, repo: SQLiteAnalyticsRepository) -> None:
        session = _make_session()
        repo.save_session(session)

        now = datetime.now(UTC)
        sessions = repo.get_sessions(now - timedelta(hours=1), now + timedelta(hours=1))

        assert sessions[0].bpm_range == session.bpm_range

    def test_tracks_played_roundtrip(self, repo: SQLiteAnalyticsRepository) -> None:
        session = _make_session()
        repo.save_session(session)

        now = datetime.now(UTC)
        sessions = repo.get_sessions(now - timedelta(hours=1), now + timedelta(hours=1))

        retrieved = sessions[0]
        assert len(retrieved.tracks_played) == 3
        assert retrieved.tracks_played[0].genre == "techno"
        assert retrieved.tracks_played[0].bpm == 130.0


class TestDateFiltering:
    def test_filter_by_date_range(self, repo: SQLiteAnalyticsRepository) -> None:
        old = _make_session(timestamp=datetime(2025, 1, 1, tzinfo=UTC))
        recent = _make_session(timestamp=datetime.now(UTC))

        repo.save_session(old)
        repo.save_session(recent)

        now = datetime.now(UTC)
        sessions = repo.get_sessions(now - timedelta(days=1), now + timedelta(hours=1))

        assert len(sessions) == 1
        assert sessions[0].session_id == recent.session_id

    def test_no_sessions_in_range(self, repo: SQLiteAnalyticsRepository) -> None:
        old = _make_session(timestamp=datetime(2025, 1, 1, tzinfo=UTC))
        repo.save_session(old)

        now = datetime.now(UTC)
        sessions = repo.get_sessions(now - timedelta(days=1), now)

        assert len(sessions) == 0


class TestAggregateStats:
    def test_aggregate_with_sessions(self, repo: SQLiteAnalyticsRepository) -> None:
        repo.save_session(_make_session())
        repo.save_session(_make_session())

        stats = repo.get_aggregate_stats(days=1)

        assert stats.total_sessions == 2
        assert stats.total_tracks_played == 6
        assert stats.avg_bpm > 0
        assert stats.avg_energy > 0
        assert len(stats.top_genres) > 0

    def test_aggregate_empty(self, repo: SQLiteAnalyticsRepository) -> None:
        stats = repo.get_aggregate_stats(days=30)

        assert stats.total_sessions == 0
        assert stats.total_tracks_played == 0
        assert stats.avg_bpm == 0.0


class TestDeleteAndCount:
    def test_delete_session(self, repo: SQLiteAnalyticsRepository) -> None:
        session = _make_session()
        repo.save_session(session)
        assert repo.count_sessions() == 1

        repo.delete_session(str(session.session_id))
        assert repo.count_sessions() == 0

    def test_count_sessions(self, repo: SQLiteAnalyticsRepository) -> None:
        assert repo.count_sessions() == 0
        repo.save_session(_make_session())
        assert repo.count_sessions() == 1
        repo.save_session(_make_session())
        assert repo.count_sessions() == 2
