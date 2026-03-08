"""Tests for SessionAnalytics entity."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from src.domain.entities.session_analytics import (
    AggregateStats,
    SessionAnalytics,
    TrackPlayEvent,
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


def _make_session(track_count: int = 5) -> SessionAnalytics:
    tracks = [_make_event(minutes_offset=i * 5) for i in range(track_count)]
    return SessionAnalytics(
        session_id=uuid.uuid4(),
        timestamp=datetime(2026, 1, 1, 20, 0, tzinfo=UTC),
        tracks_played=tracks,
        energy_curve=[0.5, 0.6, 0.7, 0.8, 0.9],
        genre_distribution={"techno": 5},
        bpm_range=(128.0, 135.0),
        key_transitions=[("Am", "Cm"), ("Cm", "Dm")],
        avg_transition_quality=0.85,
    )


class TestSessionAnalytics:
    def test_track_count(self) -> None:
        session = _make_session(5)
        assert session.track_count == 5

    def test_duration_minutes(self) -> None:
        session = _make_session(5)
        assert session.duration_minutes == 20.0

    def test_duration_single_track(self) -> None:
        session = _make_session(1)
        assert session.duration_minutes == 0.0

    def test_empty_tracks(self) -> None:
        session = SessionAnalytics(
            session_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            tracks_played=[],
            energy_curve=[],
            genre_distribution={},
            bpm_range=(0.0, 0.0),
            key_transitions=[],
            avg_transition_quality=0.0,
        )
        assert session.track_count == 0
        assert session.duration_minutes == 0.0

    def test_equality_by_session_id(self) -> None:
        sid = uuid.uuid4()
        now = datetime.now(UTC)
        s1 = SessionAnalytics(sid, now, [], [], {}, (0, 0), [], 0.0)
        s2 = SessionAnalytics(sid, now, [], [], {}, (0, 0), [], 0.0)
        assert s1 == s2

    def test_inequality(self) -> None:
        now = datetime.now(UTC)
        s1 = SessionAnalytics(uuid.uuid4(), now, [], [], {}, (0, 0), [], 0.0)
        s2 = SessionAnalytics(uuid.uuid4(), now, [], [], {}, (0, 0), [], 0.0)
        assert s1 != s2

    def test_hash_by_session_id(self) -> None:
        sid = uuid.uuid4()
        now = datetime.now(UTC)
        s1 = SessionAnalytics(sid, now, [], [], {}, (0, 0), [], 0.0)
        s2 = SessionAnalytics(sid, now, [_make_event()], [0.5], {"techno": 1}, (130, 130), [], 0.5)
        assert hash(s1) == hash(s2)
        assert len({s1, s2}) == 1

    def test_frozen(self) -> None:
        session = _make_session()
        with pytest.raises(AttributeError):
            session.avg_transition_quality = 1.0  # type: ignore[misc]


class TestTrackPlayEvent:
    def test_creation(self) -> None:
        event = _make_event(energy=0.9, genre="house", bpm=125.0, key="Cm")
        assert event.energy == 0.9
        assert event.genre == "house"
        assert event.bpm == 125.0
        assert event.key == "Cm"

    def test_frozen(self) -> None:
        event = _make_event()
        with pytest.raises(AttributeError):
            event.energy = 0.0  # type: ignore[misc]
