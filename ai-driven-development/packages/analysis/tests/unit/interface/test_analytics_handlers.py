"""Tests for analytics JSON-RPC handlers."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest

from src.application.dto.analytics_dashboard_dto import (
    AnalyticsDashboardDTO,
    EnergyReportDTO,
    GenreDistributionDTO,
    MixingPatternsDTO,
    SessionTimelineItemDTO,
)
from src.application.use_cases.generate_analytics import GenerateAnalytics
from src.application.use_cases.record_session import RecordSession
from src.interface.handlers import register_handlers
from src.interface.server import JsonRpcServer


@pytest.fixture
def server() -> JsonRpcServer:
    return JsonRpcServer()


@pytest.fixture
def mock_record_session() -> MagicMock:
    return MagicMock(spec=RecordSession)


@pytest.fixture
def mock_generate_analytics() -> MagicMock:
    return MagicMock(spec=GenerateAnalytics)


@pytest.fixture
def mock_analyze_track() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_batch_analyze() -> MagicMock:
    return MagicMock()


class TestRecordSessionHandler:
    def test_record_session_registers(
        self,
        server: JsonRpcServer,
        mock_analyze_track: MagicMock,
        mock_batch_analyze: MagicMock,
        mock_record_session: MagicMock,
        mock_generate_analytics: MagicMock,
    ) -> None:
        register_handlers(
            server,
            mock_analyze_track,
            mock_batch_analyze,
            record_session=mock_record_session,
            generate_analytics=mock_generate_analytics,
        )
        assert "record_session" in server._methods
        assert "get_analytics" in server._methods

    def test_record_session_calls_use_case(
        self,
        server: JsonRpcServer,
        mock_analyze_track: MagicMock,
        mock_batch_analyze: MagicMock,
        mock_record_session: MagicMock,
        mock_generate_analytics: MagicMock,
    ) -> None:
        register_handlers(
            server,
            mock_analyze_track,
            mock_batch_analyze,
            record_session=mock_record_session,
            generate_analytics=mock_generate_analytics,
        )

        session_data = {
            "session_id": str(uuid.uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "tracks_played": [
                {
                    "track_id": str(uuid.uuid4()),
                    "played_at": datetime.now(UTC).isoformat(),
                    "energy": 0.7,
                    "genre": "techno",
                    "bpm": 130.0,
                    "key": "Am",
                }
            ],
            "energy_curve": [0.7],
            "genre_distribution": {"techno": 1},
            "bpm_range": [128.0, 135.0],
            "key_transitions": [],
            "avg_transition_quality": 0.85,
        }

        handler = server._methods["record_session"]
        result = handler(session_data=session_data)

        assert "session_id" in result
        mock_record_session.execute.assert_called_once()

    def test_not_registered_without_deps(
        self,
        server: JsonRpcServer,
        mock_analyze_track: MagicMock,
        mock_batch_analyze: MagicMock,
    ) -> None:
        register_handlers(server, mock_analyze_track, mock_batch_analyze)
        assert "record_session" not in server._methods
        assert "get_analytics" not in server._methods


class TestGetAnalyticsHandler:
    def test_get_analytics_calls_use_case(
        self,
        server: JsonRpcServer,
        mock_analyze_track: MagicMock,
        mock_batch_analyze: MagicMock,
        mock_record_session: MagicMock,
        mock_generate_analytics: MagicMock,
    ) -> None:
        register_handlers(
            server,
            mock_analyze_track,
            mock_batch_analyze,
            record_session=mock_record_session,
            generate_analytics=mock_generate_analytics,
        )

        dashboard = AnalyticsDashboardDTO(
            energy_reports=[],
            genre_distribution=GenreDistributionDTO(genres={}, total_tracks=0),
            mixing_patterns=MixingPatternsDTO(
                most_common_transitions=[],
                preferred_bpm_range=(0.0, 0.0),
                avg_bpm=0.0,
                key_preferences=[],
                avg_transition_quality=0.0,
            ),
            session_timeline=[],
            total_sessions=0,
            total_tracks_played=0,
        )
        mock_generate_analytics.execute.return_value = dashboard

        handler = server._methods["get_analytics"]
        result = handler(days=7)

        assert result["total_sessions"] == 0
        mock_generate_analytics.execute.assert_called_once_with(days=7)

    def test_get_analytics_default_days(
        self,
        server: JsonRpcServer,
        mock_analyze_track: MagicMock,
        mock_batch_analyze: MagicMock,
        mock_record_session: MagicMock,
        mock_generate_analytics: MagicMock,
    ) -> None:
        register_handlers(
            server,
            mock_analyze_track,
            mock_batch_analyze,
            record_session=mock_record_session,
            generate_analytics=mock_generate_analytics,
        )

        dashboard = AnalyticsDashboardDTO(
            energy_reports=[],
            genre_distribution=GenreDistributionDTO(genres={}, total_tracks=0),
            mixing_patterns=MixingPatternsDTO(
                most_common_transitions=[],
                preferred_bpm_range=(0.0, 0.0),
                avg_bpm=0.0,
                key_preferences=[],
                avg_transition_quality=0.0,
            ),
            session_timeline=[],
            total_sessions=0,
            total_tracks_played=0,
        )
        mock_generate_analytics.execute.return_value = dashboard

        handler = server._methods["get_analytics"]
        handler()

        mock_generate_analytics.execute.assert_called_once_with(days=30)
