"""Tests for RecordSession use case."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.application.use_cases.record_session import RecordSession
from src.domain.entities.session_analytics import SessionAnalytics, TrackPlayEvent


def _make_session(track_count: int = 3) -> SessionAnalytics:
    tracks = [
        TrackPlayEvent(
            track_id=uuid.uuid4(),
            played_at=datetime(2026, 1, 1, 20, 0, tzinfo=UTC) + timedelta(minutes=i * 5),
            energy=0.5 + i * 0.1,
            genre="techno",
            bpm=128.0 + i,
            key="Am",
        )
        for i in range(track_count)
    ]
    return SessionAnalytics(
        session_id=uuid.uuid4(),
        timestamp=datetime(2026, 1, 1, 20, 0, tzinfo=UTC),
        tracks_played=tracks,
        energy_curve=[0.5, 0.6, 0.7],
        genre_distribution={"techno": track_count},
        bpm_range=(128.0, 130.0),
        key_transitions=[("Am", "Am")],
        avg_transition_quality=0.8,
    )


class TestRecordSession:
    def test_saves_session(self) -> None:
        repo = MagicMock()
        use_case = RecordSession(analytics_repo=repo)
        session = _make_session()

        use_case.execute(session)

        repo.save_session.assert_called_once_with(session)

    def test_skips_empty_session(self) -> None:
        repo = MagicMock()
        use_case = RecordSession(analytics_repo=repo)
        empty_session = SessionAnalytics(
            session_id=uuid.uuid4(),
            timestamp=datetime.now(UTC),
            tracks_played=[],
            energy_curve=[],
            genre_distribution={},
            bpm_range=(0.0, 0.0),
            key_transitions=[],
            avg_transition_quality=0.0,
        )

        use_case.execute(empty_session)

        repo.save_session.assert_not_called()
