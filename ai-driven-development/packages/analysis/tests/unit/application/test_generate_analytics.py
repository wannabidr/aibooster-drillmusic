"""Tests for GenerateAnalytics use case."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.application.use_cases.generate_analytics import GenerateAnalytics
from src.domain.entities.session_analytics import SessionAnalytics, TrackPlayEvent


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
    tracks: list[TrackPlayEvent] | None = None,
    genre_dist: dict[str, int] | None = None,
    energy_curve: list[float] | None = None,
    key_transitions: list[tuple[str, str]] | None = None,
    quality: float = 0.85,
) -> SessionAnalytics:
    if tracks is None:
        tracks = [
            _make_event(0, 0.5, "techno", 128.0, "Am"),
            _make_event(5, 0.7, "techno", 130.0, "Cm"),
            _make_event(10, 0.9, "house", 125.0, "Dm"),
        ]
    if genre_dist is None:
        genre_dist = {"techno": 2, "house": 1}
    if energy_curve is None:
        energy_curve = [0.5, 0.7, 0.9]
    if key_transitions is None:
        key_transitions = [("Am", "Cm"), ("Cm", "Dm")]

    bpms = [t.bpm for t in tracks] if tracks else [0.0]
    return SessionAnalytics(
        session_id=uuid.uuid4(),
        timestamp=datetime.now(UTC),
        tracks_played=tracks,
        energy_curve=energy_curve,
        genre_distribution=genre_dist,
        bpm_range=(min(bpms), max(bpms)),
        key_transitions=key_transitions,
        avg_transition_quality=quality,
    )


@pytest.fixture
def mock_repo() -> MagicMock:
    return MagicMock()


@pytest.fixture
def use_case(mock_repo: MagicMock) -> GenerateAnalytics:
    return GenerateAnalytics(analytics_repo=mock_repo)


class TestGenerateAnalytics:
    def test_empty_sessions(self, use_case: GenerateAnalytics, mock_repo: MagicMock) -> None:
        mock_repo.get_sessions.return_value = []

        result = use_case.execute(days=30)

        assert result.total_sessions == 0
        assert result.total_tracks_played == 0
        assert result.genre_distribution.total_tracks == 0
        assert result.mixing_patterns.avg_bpm == 0.0

    def test_single_session(self, use_case: GenerateAnalytics, mock_repo: MagicMock) -> None:
        session = _make_session()
        mock_repo.get_sessions.return_value = [session]

        result = use_case.execute(days=30)

        assert result.total_sessions == 1
        assert result.total_tracks_played == 3
        assert len(result.energy_reports) == 1
        assert result.energy_reports[0].peak_energy == 0.9
        assert result.energy_reports[0].valley_energy == 0.5

    def test_genre_distribution(self, use_case: GenerateAnalytics, mock_repo: MagicMock) -> None:
        session = _make_session()
        mock_repo.get_sessions.return_value = [session]

        result = use_case.execute(days=30)

        assert result.genre_distribution.genres["techno"] == 2
        assert result.genre_distribution.genres["house"] == 1

    def test_mixing_patterns(self, use_case: GenerateAnalytics, mock_repo: MagicMock) -> None:
        session = _make_session()
        mock_repo.get_sessions.return_value = [session]

        result = use_case.execute(days=30)

        assert len(result.mixing_patterns.most_common_transitions) == 2
        assert result.mixing_patterns.avg_bpm > 0
        assert result.mixing_patterns.avg_transition_quality == 0.85

    def test_session_timeline(self, use_case: GenerateAnalytics, mock_repo: MagicMock) -> None:
        session = _make_session()
        mock_repo.get_sessions.return_value = [session]

        result = use_case.execute(days=30)

        assert len(result.session_timeline) == 1
        assert result.session_timeline[0].track_count == 3
        assert result.session_timeline[0].top_genre == "techno"

    def test_multiple_sessions(self, use_case: GenerateAnalytics, mock_repo: MagicMock) -> None:
        s1 = _make_session(quality=0.8)
        s2 = _make_session(
            tracks=[_make_event(0, 0.6, "house", 122.0, "Fm")],
            genre_dist={"house": 1},
            energy_curve=[0.6],
            key_transitions=[],
            quality=0.9,
        )
        mock_repo.get_sessions.return_value = [s1, s2]

        result = use_case.execute(days=30)

        assert result.total_sessions == 2
        assert result.total_tracks_played == 4
        assert result.genre_distribution.genres["techno"] == 2
        assert result.genre_distribution.genres["house"] == 2
        assert result.mixing_patterns.avg_transition_quality == 0.85

    def test_key_preferences(self, use_case: GenerateAnalytics, mock_repo: MagicMock) -> None:
        session = _make_session()
        mock_repo.get_sessions.return_value = [session]

        result = use_case.execute(days=30)

        keys = dict(result.mixing_patterns.key_preferences)
        assert "Am" in keys
        assert "Cm" in keys

    def test_bpm_range(self, use_case: GenerateAnalytics, mock_repo: MagicMock) -> None:
        session = _make_session()
        mock_repo.get_sessions.return_value = [session]

        result = use_case.execute(days=30)

        assert result.mixing_patterns.preferred_bpm_range[0] <= result.mixing_patterns.preferred_bpm_range[1]

    def test_days_parameter_forwarded(self, use_case: GenerateAnalytics, mock_repo: MagicMock) -> None:
        mock_repo.get_sessions.return_value = []

        use_case.execute(days=7)

        call_args = mock_repo.get_sessions.call_args[0]
        from_date, to_date = call_args
        delta = to_date - from_date
        assert 6 <= delta.days <= 8
